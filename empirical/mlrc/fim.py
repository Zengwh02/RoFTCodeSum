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

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

PRINT_PROMPT = False


def build_prompt(ex, tokenizer):
    '''
    Build prompt for FIM
    '''
    prompt = ex['code']
    prompt = FIM_PREFIX + f"\n// Please complete the comment for the following method\n/**\n * {FIM_SUFFIX}\n */\n" + prompt + FIM_MIDDLE

    global PRINT_PROMPT
    if not PRINT_PROMPT:
        print(f"Example Prompt: {prompt}")
        PRINT_PROMPT = True
    return prompt


def build_result(output):
    '''
    Build result from FIM output
    '''
    answer = output.replace('\n', '').replace('\t', ' ')
    answer = re.sub(r"\s+", " ", answer)
    if '.' in answer:
        answer = answer[:answer.index('.')] + '.'
    return answer.strip()


def generate_batch(batch, model, tokenizer):
    '''
    Generate batch results
    '''
    prompts = [build_prompt(ex, tokenizer) for ex in batch]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = model.generate(prompts, sampling_params)

    outputs = [build_result(output.outputs[0].text) for output in outputs]
    return outputs


def generate_main(args):
    '''
    Main function
    '''
    model_name_or_path = args.model
    data_path = args.data
    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))

    model = LLM(model=model_name_or_path, trust_remote_code=True, gpu_memory_utilization=0.9, max_model_len=1024)

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

    results_low = []
    results_high = []
    batch_size = args.batch_size
    row_idx = 0
    batch_ex = []
    for row in tqdm(examples, total=len(examples), position=0, leave=True, desc="Generation"):
        output_file = f"{output_dir}/{row_idx}.txt"
        res = {'solution': row['ref'], 'readability': row['readability']}
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                content = file.read()
                res['result'] = content
                
                if row['readability'] >= 4:
                    results_high.append(res)
                else:
                    results_low.append(res)

            row_idx += 1
            continue

        if (row_idx + 1) % batch_size == 0:
            batch_ex.append(row)
            row_idx += 1

            batch_result = generate_batch(batch_ex, model, tokenizer)

            for i in range(len(batch_ex)):
                ex = batch_ex[i]
                if ex['readability'] >= 4:
                    results_high.append({'solution': ex['ref'], 'readability': ex['readability'], 'result':batch_result[i]})
                else:
                    results_low.append({'solution': ex['ref'], 'readability': ex['readability'], 'result':batch_result[i]})

            for i in range(len(batch_ex)):
                output_file = f"{output_dir}/{row_idx - batch_size + i}.txt"
                with open(output_file, 'w') as f:
                    if batch_result[i]:
                        f.write(batch_result[i])
            batch_ex = []
        else:
            batch_ex.append(row)
            row_idx += 1

    batch_result = generate_batch(batch_ex, model, tokenizer)

    for i in range(len(batch_ex)):
        ex = batch_ex[i]
        if ex['readability'] >= 4:
            results_high.append({'solution': ex['ref'], 'readability': ex['readability'], 'result':batch_result[i]})
        else:
            results_low.append({'solution': ex['ref'], 'readability': ex['readability'], 'result':batch_result[i]})
    for i in range(len(batch_ex)):
        output_file = f"{output_dir}/{row_idx - len(batch_ex) + i}.txt"
        with open(output_file, 'w') as f:
            if batch_result[i]:
                f.write(batch_result[i])

    high_file = args.save_dir.replace(".jsonl", "_high.jsonl")
    low_file = args.save_dir.replace(".jsonl", "_low.jsonl")

    with open(high_file, 'w') as f:
        for item in results_high:
            f.write(json.dumps(item) + '\n')

    with open(low_file, 'w') as f:
        for item in results_low:
            f.write(json.dumps(item) + '\n')

    high_readability = [ex['readability'] for ex in results_high]
    low_readability = [ex['readability'] for ex in results_low]
    print(
        f"Avg High Readability: {sum(high_readability) / len(high_readability)}; Num High Readability: {len(high_readability)}")
    print(
        f"Avg Low Readability: {sum(low_readability) / len(low_readability)}; Num Low Readability: {len(low_readability)}")


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
