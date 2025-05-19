CHECKPOINT=0
MODEL_NAME_OR_PATH="./checkpoints/ds/checkpoint-${CHECKPOINT}"
DATA="../../data/RN/RV_test.jsonl"
BATCH_SIZE=1000
OUTPUT_DIR="./FTALL${CHECKPOINT}_RV/results"
SAVE_DIR="./FTALL${CHECKPOINT}_RV/result.jsonl"

CUDA_VISIBLE_DEVICES=0 python fim.py \
  --model ${MODEL_NAME_OR_PATH} \
  --data ${DATA} \
  --batch_size ${BATCH_SIZE} \
  --output_dir ${OUTPUT_DIR} \
  --save_dir ${SAVE_DIR} \
  >FTALL${CHECKPOINT}_RV.out 2>&1 &
