import torch
import os
import argparse
from dataclasses import dataclass, field
import transformers
from torch.utils.data import Dataset
from model.GLaMM import Legion
from model.llava import conversation as conversation_lib
from transformers import CLIPImageProcessor, Trainer, TrainingArguments, AutoTokenizer
import json
from PIL import Image
from tools.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter,
    dict_to_cuda, Summary, intersectionAndUnionGPU
)
from model.SAM.utils.transforms import ResizeLongestSide
import cv2
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="LEGION Detection Training")

    # Model-specific settings
    parser.add_argument("--version", default="./checkpoints/legion_loc_exp/")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=512, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--val_batch_size", default=64, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epoch_samples", default=8000, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument('--deepspeed', action='store_true', help='Use DeepSpeed')   
    parser.add_argument('--wandb_log', action='store_true', help='Use wandb')  

    # Evaluation settings
    parser.add_argument("--val_dataset", default="CocoCapVal|RefCOCOgRegVal|RefCOCOgSegmVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)
    parser.add_argument("--data_base_train", default="", type=str)
    parser.add_argument("--data_base_test", default="", type=str)
    parser.add_argument("--train_json_file", default="", type=str)
    parser.add_argument("--test_json_file", default="", type=str)

    parser.add_argument("--save_path", default="./checkpoint/legion_cls_train", type=str)    
    return parser.parse_args()

class LegionClsDataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.args = args
        self.train = train
        if train:
            with open(args.train_json_file, 'r') as f:
                self.data = json.load(f)
        else:
            with open(args.test_json_file, 'r') as f:
                self.data = json.load(f)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        img_path = os.path.join(self.args.data_base_train if self.train else self.args.data_base_test, data_point['image_path'] if self.train else data_point['image'] )
        label = 1 if data_point['label'] == "real" else 0

        # 读取图像
        image = cv2.imread(img_path)

        # 检查图像是否有效
        if image is None or image.size == 0:
            print(f"跳过损坏图像: {img_path}")
            # 你可以选择返回一个默认值，或者选择继续处理下一个样本
            return self.__getitem__((idx + 1) % len(self.data))  # 递归调用下一个样本

        # 转换颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 准备 Global Image Encoder 的输入
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        return {"global_enc_images": global_enc_image, "cls_gt_list": label}
    
class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        inputs = inputs.copy()
        inputs['train_cls'] = True
        return super().training_step(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs['inference_cls'] = True
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        global_enc_images = inputs.get("global_enc_images")
        cls_gt_list = inputs.get("cls_gt_list")
        outputs = model(
        global_enc_images=global_enc_images,
        cls_gt_list = cls_gt_list,
        train_cls = True
        )
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    model_args["num_level_reg_features"] = 4

    model = Legion.from_pretrained(
        args.version, torch_dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + f"---- Initialized model from: {args.version} ----" + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model

def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + f"---- Initialized tokenizer from: {args.version} ----" + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer

def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + f"---- Initialized Global Image Encoder (vision tower) from: {args.vision_tower} ----" + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'prediction_head'):
            for param in model.prediction_head.parameters():
                param.requires_grad = True
            print("Only 'prediction_head' layer is set to trainable.")
        else:
            raise AttributeError("Model does not have 'prediction_head' layer.")

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + f"---- Total parameters: {total_params} ----" + '\033[0m')
        print('\033[92m' + f"---- Trainable parameters: {trainable_params} ----" + '\033[0m')

    count_parameters(model)

def load_model(args):
    print("Loading model...")
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    print("Successfully loaded model from:", args.version)
    return model, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args)

    # Prepare datasets
    train_dataset = LegionClsDataset(args, train=True)
    eval_dataset = LegionClsDataset(args, train=False)
    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        evaluation_strategy="epoch" if not args.no_eval else "no",
        logging_steps=20,
        learning_rate=args.lr,
        weight_decay=0.0,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        fp16=(args.precision == "fp16"),
        bf16=(args.precision == "bf16"),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.workers,
        logging_dir=os.path.join(args.log_base_dir, args.exp_name, "logs"),
        report_to=["wandb"] if args.wandb_log else [],
        run_name=args.exp_name,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False, # 不设置的话，会自动识别forward不需要的参数从dataset中删掉
        label_names=['cls_gt_list'], # 必须是列表，包含dataset getitem中的label的key的名字
        save_strategy="epoch",  # 每个epoch保存模型
        lr_scheduler_type="cosine",  # 余弦学习率调度
    )

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not args.no_eval else None,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(args.save_path)

if __name__ == "__main__":
    main()