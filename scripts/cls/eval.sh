python scripts/cls/eval.py \
  --deepspeed \
  --version /path/to/legion/ckpt \
  --vision_pretrained /path/to/sam/pretrained/weights \
  --exp_name 'Legion' \
  --lr 1e-3 \
  --pretrained \
  --epochs 5 \
  --batch_size 128 \
  --test_json_file "/path/to/progan/test/json" \
  --data_base_test "/path/to/progan/test/images" \


