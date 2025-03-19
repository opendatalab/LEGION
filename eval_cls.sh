python eval_cls.py \
  --deepspeed \
  --version "/mnt/petrelfs/wensiwei/LEGION/groundingLMM/checkpoint/legion_cls_train/multilayer/progan&glide_train" \
  --vision_pretrained /mnt/hwfile/opendatalab/wensiwei/checkpoint/SAM/sam_vit_h_4b8939.pth \
  --exp_name 'Legion' \
  --lr 1e-3 \
  --pretrained \
  --epochs 5 \
  --batch_size 128 \
  --epoch_samples 720119 \
  --steps_per_epoch 5626 \
  --save_path "/mnt/petrelfs/wensiwei/LEGION/groundingLMM/checkpoint/legion_cls_train" \
  --train_json_file "/mnt/petrelfs/wensiwei/LEGION/LEGION/data/progan_cls_train.json" \
  --data_base_train "/mnt/hwfile/opendatalab/bigdata_rs/datasets/CNNDetection/progan_train" \
  --test_json_file "/mnt/petrelfs/wensiwei/LEGION/LEGION/data/cnnspot.json" \
  --data_base_test "/mnt/hwfile/opendatalab/bigdata_rs/datasets/CNNDetection/CNN_synth_testset"
  # --test_json_file "/mnt/petrelfs/wensiwei/LEGION/LEGION/data/diffusion_datasets_metadata.json" \
  # --data_base_test "/mnt/hwfile/opendatalab/bigdata_rs/datasets/diffusion_datasets" \
  # --mask_validation \
  # --resume /mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/ckpt_model_last_epoch \
  # --start_epoch 5 
  #后两行是resume才有


