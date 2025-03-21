import re
import cv2
import json
import torch
import os
import argparse
from transformers import AutoTokenizer, CLIPImageProcessor
import sys
from eval.utils import grounding_image_ecoder_preprocess, mask_to_rle_pytorch, coco_encode_rle
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

def run_reflection_inference(model_path, image_path, output_dir="./output", image_size=1024, model_max_length=512, use_mm_start_end=True, conv_type="llava_v1"):

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=None, model_max_length=model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16  # By default, using bf16
    kwargs = {"torch_dtype": torch_dtype}
    model = GLaMMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, **kwargs)

    # Update model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize Global Image Encoder (CLIP)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer the model to GPU
    model = model.bfloat16().cuda()
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device="cuda")

    # Initialize Image Processor for Global Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(image_size)

    model.eval()  # Model should be in evaluation mode for inference

    # Prompt model to return grounded conversations
    instruction = 'Could you provide a detailed analysis of artifacts in this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.'

    # Create output directory if not exists already
    os.makedirs(output_dir, exist_ok=True)

    # Run inference on the single image
    result_caption, pred_masks, phrases = inference(instruction, image_path, tokenizer, model, clip_image_processor, transform)

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks[0].cpu()
    binary_pred_masks = pred_masks_tensor > 0
    uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    result_dict = {
        "image_id": os.path.basename(image_path),
        "caption": result_caption,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    return result_dict["caption"]

def inference(instructions, image_path, tokenizer, model, clip_image_processor, transform):
    # Filter out special chars
    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

    # Prepare prompt for model inference
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instructions
    if True:  # use_mm_start_end is always True
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Read and preprocess the image (Global image encoder - CLIP)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
    image_clip = image_clip.bfloat16()  # Precision is bf16 by default

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).cuda())
    image = image.bfloat16()  # Precision is bf16 by default

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    bboxes = None  # No box/region is input in GCG task

    # Generate output
    output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list, max_tokens_new=512, bboxes=bboxes)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="The path to the image for inference.")
    parser.add_argument("--legion_model_path", required=True, help="model path.")
    args = parser.parse_args()
    
    # model_path = "/path/to/your/model"
    # image_path = "/path/to/your/image.jpg"
    output_dir = "./output"
    model_path = args.legion_model_path
    image_path = args.img_path

    result = run_reflection_inference(model_path, image_path, output_dir)
    print(result)