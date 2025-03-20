export PYTHONPATH="./:$PYTHONPATH"
tag='global_step703'

python merge_lora_weights.py \
    --version /path/to/GLaMM-GranD-Pretrained \
    --weight /path/to/legion/ckpt/$tag/pytorch_model.bin \
    --save_path /path/to/legion/ckpt/$tag \
    --vision_pretrained  /path/to/sam/pretrained/weights