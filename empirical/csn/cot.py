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

PRINT_PROMPT = False

def generate_cot(ex):
    '''
    Generate COT messages
    '''
    code = tokenizer.encode(ex['code'], truncation=False, add_special_tokens=False)[:256]
    code = tokenizer.decode(code)
    code = code.replace(ex['docstring'], "", 1)

    user_prompt = f"Code:\n```python\n{code}\n```\n\n"

    questions = [
        "What is the name of the function?",
        "What are the input parameters that are being accepted by the function?",
        "What is the expected output or return value of the function?",
        "Are there any special requirements or constraints for using the function?",
        "Does the function have any additional dependencies or external requirements?",
    ]
    user_prompt += "Question:\n"
    user_prompt += "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    user_prompt += "\nPlease answer the above questions."

    messages = [{"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="your_model",
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        stream=False
    )

    messages.append({"role": response.choices[0].message.role, "content": response.choices[0].message.content})

    return messages

def build_messages(ex, cot_messages):
    '''
    Build messages for chat
    '''
    code = tokenizer.encode(ex['code'], truncation=False, add_special_tokens=False)[:256]
    code = tokenizer.decode(code)
    code = code.replace(ex['docstring'], "", 1)

    system_prompt = "You are a programming assistant skilled at understanding code and generating concise documentation."

    user_prompt = "Let's integrate the above information."
    user_prompt += "Please generate a one-line docstring for the following code that briefly describes its functionality. Only return the docstring without any additional text.\n\n"
    user_prompt += "Please generate the text in the following format, with triple quotes surrounding the content:\n\"\"\"Generated docstring.\"\"\"\n\n"
    user_prompt += f"Code:\n```python\n{code}\n```\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    cot_messages.extend(messages)

    global PRINT_PROMPT
    if not PRINT_PROMPT:
        print(f"Example Messages: {cot_messages}")
        PRINT_PROMPT = True

    return cot_messages


def build_result(output):
    '''
    Build result from chat output
    '''
    output = re.findall(r'"""(.*?)"""', output, re.DOTALL)[0]

    answer = output.replace('\n', '').replace('\t', ' ')
    answer = re.sub(r"\s+", " ", answer)
    if '.' in answer:
        answer = answer[:answer.index('.')] + ' .'
    return answer.strip()


def generate_one(ex):
    '''
    Generate one result
    '''
    cot_messages = generate_cot(ex)

    messages = build_messages(ex, cot_messages)
    # print(messages)

    response = client.chat.completions.create(
        model="your_model",
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
    return build_result(output)


def extract_reference(ex):
    '''
    Extract reference from docstring
    '''
    ref = ' '.join(ex['docstring_tokens']).replace('\n', '')
    ref = ' '.join(ref.strip().split())
    return ref


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

    results = []
    for row_idx, row in tqdm(enumerate(examples), total=len(examples),
                             position=0, leave=True,
                             desc="Generation"):
        res = {'solution': extract_reference(row)}
        output_file = f"{output_dir}/{row_idx}.txt"
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                content = file.read()
                res['result'] = content
                results.append(res)
            continue

        generate_res = generate_one(row)

        print(generate_res)
        # time.sleep(20.0)

        # print(generate_res)
        # print("*" * 50)

        with open(output_file, 'w') as f:
            if generate_res:
                f.write(generate_res)

        res['result'] = generate_res
        results.append(res)

    with open(args.save_dir, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print("Generate all over!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="data path")
    parser.add_argument('--data_num', type=int, default=100, help="data num")
    parser.add_argument('--output_dir', type=str, help="output dir")
    parser.add_argument('--save_dir', type=str, help="save dir")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
