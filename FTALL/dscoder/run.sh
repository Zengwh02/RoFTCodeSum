CUDA_VISIBLE_DEVICES=0 python ft_dscoder.py \
  --model "deepseek-ai/deepseek-coder-1.3b-base" \
  --data_list "../../data/CSN/python_train.jsonl" "../../data/UCI/UCI5_train.jsonl" "../../data/UCI/UCI10_train.jsonl" "../../data/Rename/RF_train.jsonl" "../../data/Rename/RV_train.jsonl" \
  --output_dir "./checkpoints/ds/" \
  --epochs 5 \
  --batch_size 64 \
  --micro_batch_size 64 \
  --lr 5e-5 \
  >ft_dscoder.out 2>&1 &
