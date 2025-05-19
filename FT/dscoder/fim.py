import os
import sys
import argparse
from datasets import load_from_disk, DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
import re
import json
from vllm import LLM, SamplingParams
import torch

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

FIM_BEGIN = "<｜fim▁begin｜>"
FIM_HOLE = "<｜fim▁hole｜>"
FIM_END = "<｜fim▁end｜>"


def build_prompt(ex, tokenizer):
    '''
    Build prompt
    '''
    prompt = tokenizer.encode(ex['code'], truncation=False, add_special_tokens=False)[:256]
    prompt = tokenizer.decode(prompt)
    prompt = FIM_BEGIN + prompt.replace(ex['docstring'], FIM_HOLE, 1) + FIM_END
    return prompt


def generate_batch(batch, model, tokenizer):
    '''
    Generate batch
    '''
    prompts = [build_prompt(ex, tokenizer) for ex in batch]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128, stop=[tokenizer.eos_token, tokenizer.pad_token])

    outputs = model.generate(prompts, sampling_params)

    outputs = [output.outputs[0].text for output in outputs]
    return outputs


def extract_reference(ex):
    '''
    Extract reference
    '''
    ref = ' '.join(ex['docstring_tokens']).replace('\n', '')
    ref = ' '.join(ref.strip().split())
    return ref


def generate_main(args):
    '''
    Main function
    '''
    model_name_or_path = args.model
    data_path = args.data
    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))

    model = LLM(model=model_name_or_path, dtype='bfloat16', trust_remote_code=True, gpu_memory_utilization=0.9)

    examples = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )

    output_dir = args.output_dir
    dir_ = Path(__file__).parent / output_dir
    print("Save results in {}.".format(dir_))
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    print("Start generation process")

    results = []
    batch_size = args.batch_size
    row_idx = 0
    batch_ex = []
    for row in tqdm(examples, total=len(examples), position=0, leave=True, desc="Generation"):
        output_file = f"{output_dir}/{row_idx}.txt"
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                content = file.read()
                res = {'solution': extract_reference(row), 'result': content}
                results.append(res)
            row_idx += 1
            continue

        if (row_idx + 1) % batch_size == 0:
            batch_ex.append(row)
            row_idx += 1

            batch_solution = [extract_reference(ex) for ex in batch_ex]
            batch_result = generate_batch(batch_ex, model, tokenizer)

            results.extend(
                [{'solution': solution, 'result': result} for solution, result in zip(batch_solution, batch_result)])

            for i in range(len(batch_ex)):
                output_file = f"{output_dir}/{row_idx - batch_size + i}.txt"
                with open(output_file, 'w') as f:
                    if batch_result[i]:
                        f.write(batch_result[i])
            batch_ex = []
        else:
            batch_ex.append(row)
            row_idx += 1

    batch_solution = [extract_reference(ex) for ex in batch_ex]
    batch_result = generate_batch(batch_ex, model, tokenizer)
    results.extend(
        [{'solution': solution, 'result': result} for solution, result in zip(batch_solution, batch_result)])
    for i in range(len(batch_ex)):
        output_file = f"{output_dir}/{row_idx - len(batch_ex) + i}.txt"
        with open(output_file, 'w') as f:
            if batch_result[i]:
                f.write(batch_result[i])

    with open(args.save_dir, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print("Generate all over!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path")
    parser.add_argument('--data', type=str, help="data path")
    parser.add_argument('--batch_size', type=int, help="batch size")
    parser.add_argument('--output_dir', type=str, help="output dir")
    parser.add_argument('--save_dir', type=str, help="save dir")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
