import os
import sys
import argparse
from datasets import load_from_disk, DatasetDict, Dataset, load_dataset
from pathlib import Path
from tqdm import tqdm
import re
import json
from openai import OpenAI
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

client = OpenAI(
    api_key="your_api_key",
    base_url="your_base_url",
)


def build_messages(ex):
    '''
    Build messages for chat
    '''
    code = ex['code']

    system_prompt = "You are a programming assistant skilled at understanding code and generating concise documentation."

    user_prompt = "Please generate a one-line docstring for the following code that briefly describes its functionality. Only return the docstring without any additional text.\n\n"
    user_prompt += "Please generate the text in the following format, with triple quotes surrounding the content:\n\"\"\"Generated docstring.\"\"\"\n\n"
    user_prompt += f"Code:\n```java\n{code}\n```\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def build_result(output):
    '''
    Build result from chat output
    '''
    output = re.findall(r'"""(.*?)"""', output, re.DOTALL)[0]

    answer = output.replace('\n', '').replace('\t', ' ')
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip()


def generate_one(ex, model):
    '''
    Generate one result
    '''
    messages = build_messages(ex)
    # print(messages)

    result = ""
    while not result:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            stream=False
        )

        # print(response.choices[0].message.content)
        # time.sleep(3.0)

        output = response.choices[0].message.content
        result = build_result(output)
    return result


def generate_main(args):
    '''
    Main function
    '''
    data_path = args.data
    examples = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )
    print("Load data from {}.".format(data_path))

    examples = examples.select(range(args.data_num))
    print("Data size is {}".format(args.data_num))

    output_dir = args.output_dir
    dir_ = Path(__file__).parent / output_dir
    print("Save results in {}.".format(dir_))
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    print("Start generation process")

    results_low = []
    results_high = []
    for row_idx, row in tqdm(enumerate(examples), total=len(examples),
                             position=0, leave=True,
                             desc="Generation"):
        res = {'solution': row['ref'], 'readability': row['readability']}
        output_file = f"{output_dir}/{row_idx}.txt"
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                content = file.read()
                res['result'] = content

                if row['readability'] >= 4:
                    results_high.append(res)
                else:
                    results_low.append(res)

            continue

        generate_res = generate_one(row, args.model)

        print(generate_res)
        # time.sleep(3.0)

        # print(generate_res)
        # print("*" * 50)

        with open(output_file, 'w') as f:
            if generate_res:
                f.write(generate_res)

        res['result'] = generate_res
        if row['readability'] >= 4:
            results_high.append(res)
        else:
            results_low.append(res)

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
    parser.add_argument('--model', type=str, help="model name")
    parser.add_argument('--data', type=str, help="data path")
    parser.add_argument('--data_num', type=int, default=100, help="data num")
    parser.add_argument('--output_dir', type=str, help="output dir")
    parser.add_argument('--save_dir', type=str, help="save dir")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
