CHECKPOINT=0
MODEL_NAME_OR_PATH="../FT/dscoder/checkpoints/dscoder/checkpoint-${CHECKPOINT}"
DATA="../data/CSN/python_test.jsonl"
OUTPUT_DIR="./FT${CHECKPOINT}_Origin/results"
SAVE_DIR="./FT${CHECKPOINT}_Origin/result.jsonl"

CUDA_VISIBLE_DEVICES=3 python fim_hf.py \
  --model ${MODEL_NAME_OR_PATH} \
  --data ${DATA} \
  --output_dir ${OUTPUT_DIR} \
  --save_dir ${SAVE_DIR} \
  >test_FT${CHECKPOINT}_Origin.out 2>&1 &
