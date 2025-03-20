import os
import cv2
import pdb
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
# from utils import *
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor
from eval.ddp import *
from models.LEGION.inference import custom_collate_fn, inference
from models.LEGION.model.Legion import GLaMMForCausalLM
# from model.llava import conversation as conversation_lib
# from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide

from diffusers import AutoPipelineForInpainting
import re
from collections import OrderedDict
import hpsv2
from gpt_api_prompt import gpt4o_prompt_feedback

# from FLUX_Inpainting.controlnet_flux import FluxControlNetModel
# from FLUX_Inpainting.transformer_flux import FluxTransformer2DModel
# from FLUX_Inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

# import hpsv2x

def parse_args():
    parser = argparse.ArgumentParser(description='legion inpainter')

    # Model & Data
    parser.add_argument("--hf_model_path", required=False, default=
                        # '/mnt/hwfile/opendatalab/wensiwei/legion_example/ckpt_model_ce_1.0_dice_0.5_bce_2.0/global_step633',
                        # '/mnt/hwfile/opendatalab/wensiwei/legion_example/ckpt_model_ce_1.0_dice_0.5_bce_2.0/global_step703', 
                        '/mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/checkpoint/full_data/ckpt_model_ce_1.0_dice_0.2_bce_0.4/global_step703',
                        # '/mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/checkpoint/full_data/ckpt_model_ce_1.0_dice_0.5_bce_2.0/global_step703',
                        help="The model path in huggingface format.")
    parser.add_argument('--image_dir', type=str, default='/mnt/petrelfs/kanghengrui/lmm/det-agent/models/Legion/inpaint_data')   # /mnt/hwfile/opendatalab/bigdata_rs/datasets/Legion/our_dataset/val
    parser.add_argument("--image_size", default=512, type=int, help="image size")
    parser.add_argument('--output_dir', type=str, default='./inpaint_result_0219', help="The directory to store the refinement process")

    # Basic settings
    parser.add_argument('--device', type=int, required=False, default=0)
    parser.add_argument('--alpha', type=float, required=False, default=0.5)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)

    # Inpaint related parameters
    parser.add_argument("--max_iter", required=False, type=int, default=3)

    return parser.parse_args()


def extract_explanations(result_caption, phrases):
    """
    从文本中提取每个短语的阐述内容，并返回一个有序字典，保持短语顺序。

    参数：
    result_caption (str): 包含描述的文本内容。
    phrases (list): 包含短语的列表，用于匹配每个短语对应的阐述。

    返回：
    OrderedDict: 一个按短语顺序排列的有序字典，键是短语，值是对应的阐述内容。
    """
    explanation_dict = OrderedDict()

    # 构建正则表达式模式
    for phrase in phrases:
        # 允许短语与冒号之间有空格，并且匹配下一个短语后跟冒号的部分
        pattern = re.escape(phrase) + r'\s*[:]\s*(.*?)(?=\s*' + r'|'.join([re.escape(p) + r'\s*[:]' for p in phrases]) + r'\s*|$)'
        
        # 使用正则表达式查找每个短语后的描述
        match = re.search(pattern, result_caption, re.DOTALL)
        if match:
            explanation_dict[phrase] = match.group(1).strip()
        else:
            raise NotImplementedError

    return explanation_dict


def sd_xl_inpainting_process(model, image_path, generation_prompt, max_iter, num_inference_steps, strength, tokenizer, clip_image_processor, transform, args):
    # Open the raw image and deepcopy itself
    image = Image.open(image_path)
    init_size = image.size
    inpainted_image = image.copy()

    # Initialize inpainter
    sd_xl_inpainting_pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    ).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(42)

    # FLUX
    # controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)
    # transformer = FluxTransformer2DModel.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    # )
    # flux_pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev",
    #     controlnet=controlnet,
    #     transformer=transformer,
    #     torch_dtype=torch.bfloat16
    # ).to(args.device)

    # flux_pipe.transformer.to(torch.bfloat16)
    # flux_pipe.controlnet.to(torch.bfloat16)

    # generator = torch.Generator(device=args.device).manual_seed(42)



    # Result recorder
    image_records, mask_records, par_records, explanation_records, feedback_records = [image], [], [], [], []

    # Refine iteratively
    for iter in range(max_iter):
        # Legion model inference
        print("\n---------------------------------------------")
        print(f"** Legion Prediction Iteration: {iter+1} **")
        # result_caption, pred_masks, phrases = inference(generation_prompt, image_path, clip_image_processor, args)
        result_caption, pred_masks, phrases = inference(model, generation_prompt, np.array(inpainted_image), tokenizer, clip_image_processor, transform, args)
        bool_pred_masks = pred_masks[0].cpu() > 0
        binary_pred_masks = bool_pred_masks.int()
        combined_pred_mask = torch.any(bool_pred_masks, dim=0).int()
        mask_records.append(combined_pred_mask)
        par_records.append(combined_pred_mask.float().mean().item())

        # Explanation match
        try:
            explanations = extract_explanations(result_caption=result_caption,phrases=phrases)
        except:
            return None
        explanation_records.append(explanations)

        # Resize into 1024x1024
        # inpainted_image = inpainted_image.resize((1024,1024), Image.BICUBIC)
        # resized_pred_masks = F.interpolate(binary_pred_masks.unsqueeze(1).float(), size=(1024, 1024), mode='bicubic', align_corners=False).squeeze(1)  # (N, 1024, 1024)
        # bool_pred_masks = resized_pred_masks > 0
        # binary_pred_masks = bool_pred_masks.int()

        feedbacks = {}
        
        # Refine for every single artifacts
        print(f"** SD-XL Refinement Iteration: {iter+1} **")
        for (phrase, explanation), mask in zip(explanations.items(), binary_pred_masks):
            # pdb.set_trace()
            prompt = f"Try to avoid the problem: {explanation} And make every effort to maintain the photorealism and reasonableness of the entire image, as well as consistency and coherence in style, lighting, and shadows."
            # prompt = gpt4o_prompt_feedback(prompt=f"{phrase}: {explanation}")
            feedbacks[phrase] = prompt
            # pdb.set_trace()
            # Dilation mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask.numpy().astype(np.uint8), kernel, iterations = 5)

            # sd_xl inpainting 0.1
            inpainted_image = sd_xl_inpainting_pipe(
                prompt=prompt,
                image=inpainted_image,
                mask_image=Image.fromarray((mask * 255.0).astype(np.uint8)),
                guidance_scale=8.0,
                num_inference_steps=num_inference_steps,
                strength=strength,
                generator=generator,
            ).images[0]

            # FLUX
            # inpainted_image = flux_pipe(
            #     prompt=prompt,
            #     height=inpainted_image.size[1],
            #     width=inpainted_image.size[0],
            #     control_image=inpainted_image,
            #     control_mask=Image.fromarray((mask * 255.0).astype(np.uint8)),
            #     num_inference_steps=num_inference_steps,
            #     generator=generator,
            #     controlnet_conditioning_scale=0.9,
            #     guidance_scale=3.5,
            #     negative_prompt="",
            #     true_guidance_scale=1.0
            # ).images[0]

        # Restore to original size
        # inpainted_image = inpainted_image.resize(init_size, Image.BICUBIC)
        image_records.append(inpainted_image)
        feedback_records.append(feedbacks)

    # latest refined image run Legion again
    result_caption, pred_masks, phrases = inference(model, generation_prompt, np.array(inpainted_image), tokenizer, clip_image_processor, transform, args)
    bool_pred_masks = pred_masks[0].cpu() > 0
    binary_pred_masks = bool_pred_masks.int()
    combined_pred_mask = torch.any(bool_pred_masks, dim=0).int()
    mask_records.append(combined_pred_mask)
    par_records.append(combined_pred_mask.float().mean().item())

    # Explanation match
    try: 
        explanations = extract_explanations(result_caption=result_caption,phrases=phrases)
    except:
        return None
    explanation_records.append(explanations)
    
    image_id = os.path.basename(image_path).rsplit('.',1)[0]
    save_dir = os.path.join(args.output_dir,image_id)
    # Create save directory if not exists already
    os.makedirs(save_dir, exist_ok=True)

    # Visualize the refinement process
    visualize_process(image_records=image_records, mask_records=mask_records, save_dir=save_dir)
    # HPSv2 evaluation
    evalute_hpsv2(image_path=image_path, image_records=image_records, par_records=par_records, explanation_records=explanation_records, feedback_records=feedback_records, save_dir=save_dir)

    return inpainted_image


def visualize_process(image_records, mask_records, save_dir):
    """
    可视化图像修复过程：
    - 第一行：图像变化
    - 第二行：掩码变化
    - 第三行：图像与掩码叠加变化

    参数:
        image_records (list): 图像记录列表，每个元素是PIL对象。
        mask_records (list): 掩码记录列表，每个元素是H*W的tensor。
        args (argparse.Namespace): 包含 max_iter, alpha 的参数。
    """

    assert len(image_records) == len(mask_records) == args.max_iter + 1, \
        "image_records 和 mask_records 的数量应为 args.max_iter + 1"

    # 图像叠加显示时使用的粉红色
    # pink = np.zeros((image_records[0].size[1], image_records[0].size[0], 3), dtype=np.uint8)
    # pink[:, :, 0] = 255  # 红色通道
    # pink[:, :, 2] = 255  # 蓝色通道

    # 计算原图的长宽比
    aspect_ratio = image_records[0].size[0] / image_records[0].size[1]

    # 通过长宽比动态调整figsize，保持显示区域比例
    fig_width = (args.max_iter + 1) * 3
    fig_height = 9

    # 若长宽比偏大，则调整宽度或高度
    if aspect_ratio > 1: 
        fig_width = fig_height * aspect_ratio
    else:
        fig_height = fig_width / aspect_ratio

    fig, axes = plt.subplots(3, args.max_iter + 1, figsize=(fig_width, fig_height))

    # 设置每一列的标题
    titles = ["Origin"] + [f"Iter {i}" for i in range(1, args.max_iter + 1)]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title)

    for i, (image, mask) in enumerate(zip(image_records, mask_records)):

        ###
        pink = np.zeros((image_records[i].size[1], image_records[i].size[0], 3), dtype=np.uint8)
        pink[:, :, 0] = 255  # 红色通道
        pink[:, :, 2] = 255  # 蓝色通道

        # 转换为numpy数组
        img_array = np.array(image)
        mask_array = mask.cpu().numpy().astype(np.uint8)

        # 画出图像变化
        axes[0, i].imshow(img_array)
        axes[0, i].axis('off')

        # 画出掩码变化
        axes[1, i].imshow(mask_array, cmap='gray')
        axes[1, i].axis('off')

        # 图像和掩码叠加
        # mask_expanded = mask_array[:, :, None]
        # pdb.set_trace()
        # img_with_mask = img_array * (1 - mask_expanded) + args.alpha * pink * mask_expanded + \
        #                 (1 - args.alpha) * img_array * mask_expanded

        # pdb.set_trace()
        img_with_mask = img_array * (1 - mask_array[:, :, None]) + args.alpha * pink * mask_array[:, :, None] + (1 - args.alpha) * img_array * mask_array[:, :, None]
        img_with_mask = Image.fromarray(img_with_mask.astype(np.uint8))

        axes[2, i].imshow(img_with_mask)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"iteration_process.jpg"), bbox_inches='tight', dpi=300)


def evalute_hpsv2(image_path, image_records, par_records, explanation_records, feedback_records, save_dir):
    # image_caption_path = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/Legion/batch5/val_image_descriptions_1132.json"
    # image_captions = json.load(open(image_caption_path))
    # image_name = os.path.basename(image_path)
    # evaluation_prompt = [caption for image_path, caption in image_captions.items() if image_path.endswith(image_name)]
    # assert len(evaluation_prompt) == 1
    
    evaluation_prompt = ["A colorful pair of houses on a dock, reflecting on the water with a sunset backdrop."]

    hpsv2_scores = hpsv2.score(image_records, evaluation_prompt[0], hps_version="v2.1")
    hpsv2_scores = [float(x) for x in hpsv2_scores]

    meta_data = {}
    for idx, title in enumerate(['Origin'] + [f'Iter {i}' for i in range(1, len(hpsv2_scores))]):
        if idx < len(feedback_records):
            info_template = {
                "explanation": explanation_records[idx],
                "feedback": feedback_records[idx],
                "hpsv2": hpsv2_scores[idx],
                "par": par_records[idx]
            }
        else:
            info_template = {
                "explanation": explanation_records[idx],
                "hpsv2": hpsv2_scores[idx],
                "par": par_records[idx]
            }
        meta_data[title] = info_template

    # Save scores as json files
    with open(os.path.join(save_dir,f"meta_data.json"),'w') as fsave:
        fsave.write(json.dumps(meta_data,indent=4))

    # Plot the hpsv2 score varying curve
    iterations = list(range(len(hpsv2_scores)))  # [0, 1, 2, ..., N]
    
    # 直接使用原始的 hpsv2_scores 作为 Y 轴数值
    y_min = max(min(hpsv2_scores) - 0.01, 0)  # 留出一点下边界空间
    y_max = min(max(hpsv2_scores) + 0.01, 1)  # 留出一点上边界空间
    
    # 计算Origin得分和最高得分的差值
    origin_score = hpsv2_scores[0]
    max_score = max(hpsv2_scores)
    score_diff = max_score - origin_score

    # Define line color
    line_color = 'blue'  # 设置曲线颜色

    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot a line connecting the scores
    plt.plot(iterations, hpsv2_scores, color=line_color, linestyle='-', linewidth=2, marker='o', markersize=8)

    # 添加两条水平虚线
    plt.axhline(y=origin_score, color='black', linestyle='--', linewidth=1.5)  # Origin 水平线
    plt.axhline(y=max_score, color='black', linestyle='--', linewidth=1.5)  # 最高得分 水平线

    # 在横轴最右侧标注差值
    text_x = len(hpsv2_scores) - 0.9  # 设定在横轴最右侧
    text_y = (origin_score + max_score) / 2  # 设定在两条虚线的中间
    plt.text(text_x, text_y, f'Δ = {score_diff:.3f}', color='black', fontsize=16, ha='right', va='center',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Highlight the maximum score
    max_score_idx = np.argmax(hpsv2_scores)
    
    for i, score in enumerate(hpsv2_scores):
        # Set color for each point
        if i == 0:
            # Origin: 空心白色圆，轮廓颜色与曲线颜色一致
            plt.scatter(i, score, edgecolor=line_color, facecolor='white', linewidth=2, s=100, zorder=5)
        elif i == max_score_idx:
            # Maximum score: 红色填充圆
            plt.scatter(i, score, edgecolor='red', facecolor='red', linewidth=2, s=100, zorder=5)
        else:
            # 其他 Iteration: 填充颜色和曲线颜色一致
            plt.scatter(i, score, edgecolor=line_color, facecolor=line_color, linewidth=2, s=100, zorder=5)

    # Labels and titles
    plt.title('HPSv2 Score Progression over Inpainting Iterations', fontsize=14)
    plt.xlabel('Inpainting Iteration', fontsize=12)
    plt.ylabel('HPSv2 Score', fontsize=12)
    plt.xticks(iterations, ['Origin'] + [f'Iter {i}' for i in range(1, len(hpsv2_scores))])
    
    # 设置 y 轴范围，使得图像更清晰
    plt.ylim(y_min, y_max)
    
    # Grid and limits
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'hpsv2_score_curve.jpg'))



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
    model = GLaMMForCausalLM.from_pretrained(args.hf_model_path, low_cpu_mem_usage=True,
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
    vision_tower.to(device=args.device)

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()  # Model should be in evaluation mode for inference

    # Prompt model to return grounded conversations
    # generation_prompt = 'Could you provide a detailed analysis of artifacts in this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.'
    generation_prompt = 'Please provide a detailed analysis of artifacts in this photo, considering physical artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), structural artifacts (e.g., deformed objects, asymmetry, or distorted text), and distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation). Output with interleaved segmentation masks for the corresponding parts of the answer.'
    ### TO DO: UPDATE THE PROMPT TO ADD MORE DETAILS

    # Create DDP Dataset
    dataset = GCGEvalDDP(args.image_dir)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=2,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)
    
    # Iterate over all the images, run inference and save results
    for (image_id, image_path) in tqdm(dataloader):
        image_id, image_path = image_id[0], image_path[0]

        # raw_image = cv2.imread(image_path)
        # h, w = raw_image.shape[:2]

        # image_id = os.path.basename(image_path).rsplit('.',1)[0]
        # save_dir = os.path.join(args.output_dir,image_id)
        # if os.path.exists(save_dir):
        #     continue

        inpainted_result = sd_xl_inpainting_process(model=model, image_path=image_path, generation_prompt=generation_prompt, max_iter=args.max_iter, num_inference_steps=25, 
                                                      strength=0.84, tokenizer=tokenizer, clip_image_processor=clip_image_processor, transform=transform, args=args)

        
