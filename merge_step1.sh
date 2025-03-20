tag='global_step703'
python /path/to/legion/ckpt/zero_to_fp32.py \
       /path/to/legion/ckpt \
       /path/to/legion/ckpt/$tag/pytorch_model.bin \
       --tag "$tag"
 