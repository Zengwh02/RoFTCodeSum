import os
import sys
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
import json
import torch
from metric import calculate_bleu, calculate_sbert

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

FIM_BEGIN = "<｜fim▁begin｜>"
FIM_HOLE = "<｜fim▁hole｜>"
FIM_END = "<｜fim▁end｜>"


def build_prompt(ex, tokenizer):
    prompt = tokenizer.encode(ex['code'], truncation=False, add_special_tokens=False)[:256]
    prompt = tokenizer.decode(prompt)
    prompt = FIM_BEGIN + prompt.replace(ex['docstring'], FIM_HOLE, 1) + FIM_END
    return prompt


def generate_one(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = decoded_output[len(prompt):].strip()
    return result


def extract_reference(ex):
    ref = ' '.join(ex['docstring_tokens']).replace('\n', '')
    ref = ' '.join(ref.strip().split())
    return ref


def generate_main(args):
    model_name_or_path = args.model
    data_path = args.data
    print("model", model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    print("load model over.")

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
    row_idx = 0

    generated_results = []
    true_results = []
    bleu_score = 0
    sbert_score = 0
    count = 0
    tmp_dir = args.output_dir.rsplit('/', 1)[0] + '/tmp'

    for row in tqdm(examples, total=len(examples), position=0, leave=True, desc="Generation"):
        output_file = f"{output_dir}/{row_idx}.txt"
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                result = file.read()
        else:
            prompt = build_prompt(row, tokenizer)
            result = generate_one(prompt, model, tokenizer)
            with open(output_file, 'w') as f:
                if result:
                    f.write(result)

        solution = extract_reference(row)
        results.append({'solution': solution, 'result': result})

        generated_results.append(result)
        true_results.append(solution)

        if len(generated_results) == 100:
            batch_bleu = calculate_bleu(
                predictions=generated_results,
                references=true_results,
                dir_=tmp_dir
            )
            batch_sbert = calculate_sbert(true_results, generated_results)

            bleu_score += batch_bleu * len(generated_results)
            sbert_score += batch_sbert * len(generated_results)
            count += len(generated_results)

            avg_bleu = round(bleu_score / count, 2)
            avg_sbert = round(sbert_score / count, 2)

            print(f"Current scores - BLEU: {avg_bleu}, SBERT: {avg_sbert}")

            generated_results = []
            true_results = []
            torch.cuda.empty_cache()

        row_idx += 1

    if generated_results:
        batch_bleu = calculate_bleu(
            predictions=generated_results,
            references=true_results,
            dir_=tmp_dir
        )
        batch_sbert = calculate_sbert(true_results, generated_results)

        bleu_score += batch_bleu * len(generated_results)
        sbert_score += batch_sbert * len(generated_results)
        count += len(generated_results)

        avg_bleu = round(bleu_score / count, 2)
        avg_sbert = round(sbert_score / count, 2)

        print(f"Final scores - BLEU: {avg_bleu}, SBERT: {avg_sbert}")

    print("Generate all over!!!")

    with open(args.save_dir, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


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
