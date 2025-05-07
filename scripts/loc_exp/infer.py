import re
import cv2
import json
import bleach
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor

from eval.utils import *
from eval.ddp import *
from model.Legion import LegionForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="LEGION Inference")

    parser.add_argument("--hf_model_path", required=True, help="The directory to your legion ckpt")
    parser.add_argument("--image_dir", required=False, default="./inference_data", help="The directory containing images to run inference.")
    parser.add_argument("--output_dir", required=True, help="The directory to store the inference result.")

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)

    return parser.parse_args()


def inference(model, instructions, image_np, tokenizer, clip_image_processor, transform, args):
    # Filter out special chars
    instructions = bleach.clean(instructions)
    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

    # Prepare prompt for model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instructions
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
    image_clip = image_clip.bfloat16()  # Precision is bf16 by default

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)  # 不要注释
    # image = image_np
    resize_list = [image.shape[:2]]
    image = (
        grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
    image = image.bfloat16()  # Precision is bf16 by default

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    bboxes = None  # No box/region is input in GCG task

    # Generate output
    output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                            max_tokens_new=512, bboxes=bboxes)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    # Post-processing
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()
    return cleaned_str, pred_masks, phrases


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    image_path = [item[1] for item in batch]

    return image_id, image_path



if __name__ == "__main__":

    args = parse_args()
    init_distributed_mode(args)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, cache_dir=None,
                                              model_max_length=args.model_max_length, padding_side="right",
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16  # By default, using bf16
    kwargs = {"torch_dtype": torch_dtype}
    model = LegionForCausalLM.from_pretrained(args.hf_model_path, low_cpu_mem_usage=True,
                                             seg_token_idx=seg_token_idx, **kwargs)
    # Update model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize Global Image Encoder (CLIP)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer the model to GPU
    model = model.bfloat16().cuda()  # Replace with model = model.float().cuda() for 32 bit inference
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device="cuda")

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()  

    # Prompt model to return grounded conversations
    instruction = 'Please provide a detailed analysis of artifacts in this photo, considering physical artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), structural artifacts (e.g., deformed objects, asymmetry, or distorted text), and distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation). Output with interleaved segmentation masks for the corresponding parts of the answer.'

    # Create DDP Dataset
    img_dir = args.image_dir
    dataset = GCGEvalDDP(img_dir)
    distributed_sampler = DistributedSampler(dataset, rank=args.local_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=2,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)
    
    # Iterate over all the images, run inference and save results
    for (image_id, image_path) in tqdm(dataloader):

        image_id, image_path = image_id[0], image_path[0]
        raw_img = cv2.imread(image_path)

        result_caption, pred_masks, phrases = inference(model, instruction, raw_img.astype(np.uint8), tokenizer, clip_image_processor, transform, args)  # GLaMM Inference

        pred_masks_tensor = pred_masks[0].cpu()
        binary_pred_masks = pred_masks_tensor > 0
        pred_mask = torch.any(binary_pred_masks, dim=0).int()

        # save
        save_dir = os.path.join(args.output_dir,image_id)
        os.makedirs(save_dir,exist_ok=True)

        # explanation results
        explanation_res = {
            image_path: result_caption
        }
        with open(os.path.join(save_dir,"explanation.json"), 'w') as f:
            json.dump(explanation_res, f, indent=4)

        # localization results
        pred_mask = (pred_mask.numpy() * 255).astype(np.uint8)
        pred_mask = Image.fromarray(pred_mask, mode='L')
        pred_mask.save(os.path.join(save_dir, "mask.png"))