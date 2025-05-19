CUDA_VISIBLE_DEVICES=0 python clawsat_dscoder.py \
  --model "deepseek-ai/deepseek-coder-1.3b-base" \
  --data_a "../../data/CSN/python_train.jsonl" \
  --data_b "../../data/Rename/RF_train.jsonl" \
  --data_c "../../data/Rename/RV_train.jsonl" \
  --output_dir "./checkpoints/RN_dscoder/" \
  --epochs 3 \
  --batch_size 64 \
  --micro_batch_size 64 \
  --lr 5e-5 \
  >RN_clawsat_dscoder.out 2>&1 &
