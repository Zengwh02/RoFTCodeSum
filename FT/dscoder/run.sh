CUDA_VISIBLE_DEVICES=2 python ft_dscoder.py \
  --model "deepseek-ai/deepseek-coder-1.3b-base" \
  --data "../../data/CSN/python_train.jsonl" \
  --output_dir "./checkpoints/dscoder/" \
  --epochs 3 \
  --batch_size 64 \
  --micro_batch_size 64 \
  --lr 5e-5 \
  >ft_dscoder.out 2>&1 &
