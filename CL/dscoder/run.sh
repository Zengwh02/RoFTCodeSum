CUDA_VISIBLE_DEVICES=0 python cl_dscoder.py \
  --model "deepseek-ai/deepseek-coder-1.3b-base" \
  --data_a "../../data/CSN/python_train.jsonl" \
  --data_b "../../data/Rename/RF_train.jsonl" \
  --data_c "../../data/Rename/RV_train.jsonl" \
  --output_dir "./checkpoints/RN_dscoder/" \
  --epochs_per_data 1 \
  --batch_size 64 \
  --micro_batch_size 64 \
  --lr 5e-5 \
  >RN_cl_dscoder.out 2>&1 &
