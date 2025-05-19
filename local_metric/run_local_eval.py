import json
import glob
from metric import calculate_bleu, calculate_sbert, calculate_meteor, calculate_rouge_l
import numpy as np
from tqdm import tqdm
import os
import sys
import re

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.setrecursionlimit(10000)  # 设置最大递归深度为10000


def get_jsonl_file_list(input_file_or_dir):
    '''
    Get list of jsonl files
    '''
    jsonl_file_list = []
    if os.path.isdir(input_file_or_dir):
        jsonl_file_list += glob.glob(os.path.join(input_file_or_dir, "*.jsonl"))
    elif input_file_or_dir.endswith(".jsonl"):
        jsonl_file_list.append(input_file_or_dir)
    else:
        raise ValueError("入参错误")
    return jsonl_file_list


def evaluate_batch(batch_samples, tmp_dir):
    '''
    Evaluate batch samples
    '''
    solutions = [sample['solution'] for sample in batch_samples]
    results = [sample['result'] for sample in batch_samples]

    batch_bleu = calculate_bleu(results, solutions, tmp_dir)
    batch_sbert = calculate_sbert(solutions, results)
    batch_meteor = calculate_meteor(results, solutions)
    batch_rouge_l = calculate_rouge_l(results, solutions)

    return batch_bleu, batch_sbert, batch_meteor, batch_rouge_l


def main(input_file_or_dir=""):
    '''
    Main function
    '''
    jsonl_file_list = get_jsonl_file_list(input_file_or_dir)
    for jsonl_file in jsonl_file_list:
        with open(jsonl_file, "r", encoding="utf8") as f:
            content = [json.loads(i) for i in f.readlines()]
        batch_size = 100
        total_bleu = 0
        total_sbert = 0
        total_meteor = 0
        total_rouge_l = 0

        total_samples = len(content)
        num_full_batches = total_samples // batch_size
        remaining_samples = total_samples % batch_size

        for i in tqdm(range(num_full_batches), desc=f"Evaluating {jsonl_file}"):
            start_idx = i * batch_size
            batch = content[start_idx:start_idx + batch_size]
            tmp_dir = input_file_or_dir.rsplit('/', 1)[0] + '/tmp'

            batch_bleu, batch_sbert, batch_meteor, batch_rouge_l = evaluate_batch(batch, tmp_dir)

            total_bleu += batch_bleu * batch_size
            total_sbert += batch_sbert * batch_size
            total_meteor += batch_meteor * batch_size
            total_rouge_l += batch_rouge_l * batch_size

            print(total_bleu / ((i + 1) * batch_size))

        if remaining_samples > 0:
            start_idx = num_full_batches * batch_size
            final_batch = content[start_idx:]
            tmp_dir = input_file_or_dir.rsplit('/', 1)[0] + '/tmp'

            batch_bleu, batch_sbert, batch_meteor, batch_rouge_l = evaluate_batch(final_batch, tmp_dir)

            total_bleu += batch_bleu * remaining_samples
            total_sbert += batch_sbert * remaining_samples
            total_meteor += batch_meteor * remaining_samples
            total_rouge_l += batch_rouge_l * remaining_samples

        avg_bleu = total_bleu / total_samples
        avg_sbert = total_sbert / total_samples
        avg_meteor = total_meteor / total_samples
        avg_rouge_l = total_rouge_l / total_samples

        score_info = {
            "from": "data",
            "bleu": avg_bleu,
            "sbert": avg_sbert,
            "meteor": avg_meteor,
            "rouge_l": avg_rouge_l,
            "num": total_samples,
        }
        print(json.dumps(score_info, indent=4))
        return score_info


if __name__ == '__main__':
    print('Response score=>')
    file_name = 'result_file_name'
    info, _ = main(input_file_or_dir=file_name)
    with open("eval_file_name", 'w', encoding='utf-8') as f:
        f.write(json.dumps(info))

