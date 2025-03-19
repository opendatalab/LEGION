import torch
import os
import wandb
import argparse
from dataclasses import dataclass, field
import transformers
from torch.utils.data import Dataset, DataLoader
from model.GLaMM import Legion
from model.llava import conversation as conversation_lib
from transformers import CLIPImageProcessor
import pdb
import json
from PIL import Image
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU)
from model.SAM.utils.transforms import ResizeLongestSide
import cv2
from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# os.environ["WANDB_RESUME"] = "allow"
# os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Legion Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
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

    parser.add_argument("--save_path", default="/mnt/petrelfs/wensiwei/LEGION/groundingLMM/checkpoint/legion_cls_train", type=str)    
    return parser.parse_args()

class legion_cls_dataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.args = args
        self.train = train
        if train == True:
            with open(args.train_json_file, 'r') as f:
                self.data = json.load(f)
        elif train == False:
            with open(args.test_json_file, 'r') as f:
                self.data = json.load(f)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train == True:
            img_path = os.path.join(self.args.data_base_train, self.data[idx]['image'])
        else:
            img_path = os.path.join(self.args.data_base_test, self.data[idx]['image'])
        label = 1 if self.data[idx]['label'] == "real" else 0
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cate = self.data[idx]['category']
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return global_enc_image, label, img_path, cate
    



def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    model_args["num_level_reg_features"] = 4

    model = Legion.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model

def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '/mnt/hwfile/opendatalab/wensiwei/checkpoint/GLaMM-GranD-Pretrained', model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
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
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.parameters():
            param.requires_grad = False



    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)

# change this function to our own to support custom behaviors
def load_model(args):
    print("Loading model...")
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    print("Successfully loaded model from:", args.version)
    return model, tokenizer



def validate(args, model, cls_test_dataloader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    with torch.no_grad():
        for images, labels, _, cates in tqdm(cls_test_dataloader):
            images = images.to(device)
            logits = model(global_enc_images=images, inference_cls=True)['logits'].cpu()
            _, pred_cls = torch.max(logits, dim=1)
            for label, pred, cate in zip(labels.tolist(), pred_cls.tolist(), cates):
                if cate not in results:
                    results[cate] = {'right':0, 'wrong':0}
                if label == pred:
                    results[cate]['right'] += 1
                else:
                    results[cate]['wrong'] += 1
    acc_sum = 0
    acc_num = 0
    for cate, result in results.items():
        result['acc'] = result['right'] / (result['right'] + result['wrong'])
        acc_sum += result['acc']
        acc_num += 1
        print(cate, result)
    print(f'avg: {acc_sum/acc_num}')
            # acc_num += (pred_cls == labels).sum().item()
            # total_num += images.shape[0]
        # print(f"acc_num/total_num:{acc_num}/{total_num}, acc:{acc_num/total_num}")
    # return acc_num / total_num
        
    
def main():
    args = parse_args()
    model, processor = load_model(args)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    cls_test_dataset = legion_cls_dataset(args, train=False)
    cls_test_dataloader = DataLoader(
        cls_test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    validate(args, model, cls_test_dataloader)

    


if __name__ == "__main__":
    main()
    
    