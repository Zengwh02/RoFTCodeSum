import torch
from torch.utils.data import DataLoader
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Sequence
import transformers
from datasets import load_dataset, Dataset
import copy
from tqdm import tqdm
import argparse
import pprint
import sys
from dataclasses import dataclass

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

IGNORE_INDEX = -100
FIM_BEGIN = "<｜fim▁begin｜>"
FIM_HOLE = "<｜fim▁hole｜>"
FIM_END = "<｜fim▁end｜>"


def tokenize_dataset(data_path, tokenizer):
    '''
    Tokenize dataset
    '''
    file_name = os.path.basename(data_path)
    tokenized_data = f"../../data/dscoder_tokenized_{file_name}"
    if os.path.exists(tokenized_data):
        ds = load_dataset('json', data_files=tokenized_data, split="train")
        return ds
    else:
        # build prompt using fill-in-the-middle
        def build_prompt(code, docstring):
            '''
            Build prompt
            '''
            prompt = tokenizer.encode(code, truncation=False, add_special_tokens=False)[:256]
            prompt = tokenizer.decode(prompt)
            prompt = FIM_BEGIN + prompt.replace(docstring, FIM_HOLE, 1) + FIM_END
            return prompt

        def extract_reference(docstring_tokens):
            '''
            Extract reference
            '''
            ref = ' '.join(docstring_tokens).replace('\n', '')
            ref = ' '.join(ref.strip().split())
            ref = tokenizer.encode(ref, truncation=False, add_special_tokens=False)[:128]
            ref = tokenizer.decode(ref)
            return ref

        def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
            '''
            Tokenize a list of strings
            '''
            tokenized_list = [
                tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=400,
                    truncation=True,
                )
                for text in strings
            ]

            input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
            input_ids_lens = labels_lens = [
                tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
            ]

            return dict(
                input_ids=input_ids,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens,
            )

        def preprocess(
                sources: Sequence[str],
                targets: Sequence[str],
        ) -> Dict:
            '''
            Preprocess
            '''
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in
                                                     (examples, sources)]

            input_ids = examples_tokenized["input_ids"]

            labels = copy.deepcopy(input_ids)
            for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
                label[:source_len] = IGNORE_INDEX
            return dict(input_ids=input_ids, labels=labels)

        def train_tokenize_function(examples):
            '''
            Train tokenize function
            '''
            codes = examples['code']
            docstrings = examples['docstring']
            docstring_tokens = examples['docstring_tokens']

            sources = [build_prompt(c, d) for c, d in zip(codes, docstrings)]
            targets = [extract_reference(d) for d in docstring_tokens]

            data_dict = preprocess(sources, targets)
            return data_dict

        ds: Dataset = load_dataset('json', data_files=data_path, split="train")

        tokenized_dataset = ds.map(
            train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=10,
            remove_columns=ds.column_names,
            desc="Running Encoding",
        )

        tokenized_dataset.to_json(tokenized_data)

    return tokenized_dataset


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.stack([torch.tensor(x) for x in input_ids])
        labels = torch.stack([torch.tensor(x) for x in labels])

        return dict(
            input_ids=input_ids,
            labels=labels,
        )


def run_FT(args, model, tokenizer, data, optimizer, scheduler):
    '''
    Run FT
    '''
    grad_accumulation = int(args.batch_size / args.micro_batch_size)
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(data, total=len(data), position=0, leave=True, desc="Training Progress"):
            step_loss = 0.0
            optimizer.zero_grad()
            for i in range(0, len(batch['input_ids']), args.micro_batch_size):
                input_ids = batch['input_ids'][i:i + args.micro_batch_size].to(model.device)
                labels = batch['labels'][i:i + args.micro_batch_size].to(model.device)

                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accumulation
                loss.backward()

                step_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]
            print(f"Loss: {step_loss:.4f}, LR: {current_lr:.2e}")

        os.makedirs(f"{args.output_dir}checkpoint-{epoch}", exist_ok=True)
        model.save_pretrained(f"{args.output_dir}checkpoint-{epoch}")
        tokenizer.save_pretrained(f"{args.output_dir}checkpoint-{epoch}")


def main(args):
    '''
    Main function
    '''
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    model_name = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        trust_remote_code=True
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_loader = DataLoader(tokenize_dataset(args.data, tokenizer), batch_size=args.batch_size,
                             shuffle=True, collate_fn=data_collator, num_workers=4)

    optimizer = transformers.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    num_training_steps = len(data_loader) * args.epochs
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps
    )

    run_FT(args, model, tokenizer, data_loader, optimizer, scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="deepseek-ai/deepseek-coder-1.3b-base", type=str, help="model name or path")
    parser.add_argument('--data', default="", type=str, help="data")
    parser.add_argument('--output_dir', default="", type=str, help="output_dir")
    parser.add_argument('--epochs', default=1, type=int, help="num_train_epochs")
    parser.add_argument('--batch_size', default=64, type=int, help="batch_size")
    parser.add_argument('--micro_batch_size', default=1, type=int, help="micro_batch_size")
    parser.add_argument('--lr', default=2e-5, type=float, help="global learning rate")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
