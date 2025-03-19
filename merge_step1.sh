#!/bin/bash

# tag='global_step720'
# python /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/ckpt_model_ce_1.0_dice_4.0_bce_2.0/zero_to_fp32.py \
#     /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/ckpt_model_ce_1.0_dice_4.0_bce_2.0 \
#     /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/ckpt_model_ce_1.0_dice_4.0_bce_2.0/$tag/pytorch_model.bin \
#     --tag "$tag"

# 正常output
# tag='global_step703'
# python /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/ckpt_model_ce_1.0_dice_0.1_bce_0.4/zero_to_fp32.py \
#     /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/ckpt_model_ce_1.0_dice_0.1_bce_0.4 \
#     /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output/Legion/ckpt_model_ce_1.0_dice_0.1_bce_0.4/$tag/pytorch_model.bin \
#     --tag "$tag"

tag='global_step703'
python /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output_origin_single_prompt/Legion/ckpt_model_ce_1.0_dice_0.2_bce_0.4/zero_to_fp32.py \
    /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output_origin_single_prompt/Legion/ckpt_model_ce_1.0_dice_0.2_bce_0.4 \
    /mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/output_origin_single_prompt/Legion/ckpt_model_ce_1.0_dice_0.2_bce_0.4/$tag/pytorch_model.bin \
    --tag "$tag"
 