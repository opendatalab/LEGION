# export PYTHONPATH="./:$PYTHONPATH"
# tag='global_step720'

# python scripts/merge_lora_weights.py \
#     --version /mnt/hwfile/opendatalab/wensiwei/checkpoint/GLaMM-GranD-Pretrained \
#     --weight /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/ckpt_model_ce_1.0_dice_4.0_bce_2.0/$tag/pytorch_model.bin \
#     --save_path /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/checkpoint/ckpt_model_ce_1.0_dice_4.0_bce_2.0/$tag \
#     --vision_pretrained /mnt/hwfile/opendatalab/wensiwei/checkpoint/SAM/sam_vit_h_4b8939.pth 

# 正常output
# export PYTHONPATH="./:$PYTHONPATH"
# tag='global_step703'
# setting='ckpt_model_ce_1.0_dice_0.1_bce_0.4'

# python scripts/merge_lora_weights.py \
#     --version /mnt/hwfile/opendatalab/wensiwei/checkpoint/GLaMM-GranD-Pretrained \
#     --weight /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/$setting/$tag/pytorch_model.bin \
#     --save_path /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/checkpoint/full_data/$setting/$tag \
#     --vision_pretrained  /mnt/hwfile/opendatalab/wensiwei/checkpoint/SAM/sam_vit_h_4b8939.pth


export PYTHONPATH="./:$PYTHONPATH"
tag='global_step703'
setting='ckpt_model_ce_1.0_dice_0.2_bce_0.4'

python scripts/merge_lora_weights.py \
    --version /mnt/hwfile/opendatalab/wensiwei/checkpoint/GLaMM-GranD-Pretrained \
    --weight /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output_origin_single_prompt/Legion/$setting/$tag/pytorch_model.bin \
    --save_path /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/checkpoint/full_data/origin_single_prompt/$setting/$tag \
    --vision_pretrained  /mnt/hwfile/opendatalab/wensiwei/checkpoint/SAM/sam_vit_h_4b8939.pth