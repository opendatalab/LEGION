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
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from PIL import Image
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - GCG")

    parser.add_argument("--hf_model_path", required=True, help="The model path in huggingface format.")
    parser.add_argument("--img_dir", required=False, default="./data/GranDf/GranDf_HA_images/val_test",
                        help="The directory containing images to run inference.")
    parser.add_argument("--output_dir", required=True, help="The directory to store the response in json format.")

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--output_path', default='env://', help='url used to set up distributed training')
    parser.add_argument('--gt_path', default='')

    parser.add_argument("--apply_jpeg", action="store_true", default=False)
    parser.add_argument('--jpeg_quality', default=50, type=int)

    parser.add_argument("--apply_gaussian_noise", action="store_true", default=False)
    parser.add_argument('--gaussian_noise_std', default=25.5, type=float)

    parser.add_argument("--apply_gaussian_blur", action="store_true", default=False)
    parser.add_argument('--gaussian_blur_size', default=5, type=int)

    parser.add_argument('--dataset', required=True, type=str)


    return parser.parse_args()




 


def calculate_miou(pred: torch.tensor, label: torch.tensor) -> float:
    """
    计算两个二值[h, w]张量的mIoU。

    参数:
        pred (torch.Tensor): 预测二值张量，形状为[h, w],值为0或1。
        label (torch.Tensor): 标签二值张量，形状为[h, w],值为0或1。

    返回:
        float: mIoU值。
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)

    ious = []

    for cls in [0, 1]:
        pred_mask = (pred == cls)
        label_mask = (label == cls)

        intersection = torch.sum(pred_mask & label_mask).item()
        union = torch.sum(pred_mask | label_mask).item()

        iou = intersection / (union + 1e-6)
        ious.append(iou)
    miou = sum(ious) / len(ious)

    return miou

def calculate_miou_without_background(pred: torch.tensor, label: torch.tensor) -> float:
    """
    计算两个二值[h, w]张量的mIoU。

    参数:
        pred (torch.Tensor): 预测二值张量，形状为[h, w],值为0或1。
        label (torch.Tensor): 标签二值张量，形状为[h, w],值为0或1。

    返回:
        float: mIoU值。
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)

    ious = []

    for cls in [1]:
        pred_mask = (pred == cls)
        label_mask = (label == cls)

        intersection = torch.sum(pred_mask & label_mask).item()
        union = torch.sum(pred_mask | label_mask).item()

        iou = intersection / (union + 1e-6)
        ious.append(iou)
    miou = sum(ious) / len(ious)

    return miou

def calculate_f1(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    计算二值分割任务的F1分数。

    参数:
        pred (torch.Tensor): 预测二值张量，形状为 [h, w]，值为0或1。
        label (torch.Tensor): 标签二值张量，形状为 [h, w]，值为0或1。

    返回:
        float: F1分数，范围在0到1之间。
    """
    if pred.shape != label.shape:
        raise ValueError(f"Shape mismatch: pred shape {pred.shape} and label shape {label.shape} must be the same.")

    # 验证输入是否为二值
    if not torch.all((pred == 0) | (pred == 1)):
        raise ValueError("Pred tensor must be binary (0 or 1).")
    if not torch.all((label == 0) | (label == 1)):
        raise ValueError("Label tensor must be binary (0 or 1).")

    # 将预测和标签转换为布尔类型
    pred = pred.bool()
    label = label.bool()

    # 计算 True Positives (TP), False Positives (FP), 和 False Negatives (FN)
    TP = torch.sum(pred & label).item()
    FP = torch.sum(pred & ~label).item()
    FN = torch.sum(~pred & label).item()

    # 计算精确率（Precision）和召回率（Recall）
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算F1分数，处理特殊情况
    if TP == 0 and FP == 0 and FN == 0:
        f1 = 1.0
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1



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

    # Read and preprocess the image (Global image encoder - CLIP)
    # image_np = cv2.imread(image_path)
    ###TODO:这个地方后续需要去掉
    # image_np = cv2.resize(image_np, (512, 512))
    ###
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


def visualize(pred_mask, gt_mask, name):

    # 转换为 NumPy 数组，如果是 PyTorch tensor
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.numpy()

    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.numpy()

    # 创建一个subplot，2行1列的布局，绘制两个掩码
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制 pred_mask
    ax[0].imshow(pred_mask, cmap='gray')
    ax[0].set_title('Pred Mask')
    ax[0].axis('off')  # 不显示坐标轴

    # 绘制 gt_mask
    ax[1].imshow(gt_mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')  # 不显示坐标轴

    # 调整布局
    plt.tight_layout()

    # 设置保存路径
    save_path = f'/mnt/petrelfs/wensiwei/LEGION/groundingLMM/eval_result/image/mask_{name}.jpg'

    # 保存图片
    plt.savefig(save_path)

    # 关闭 plt，释放资源
    plt.close()



if __name__ == "__main__":

    args = parse_args()
    init_distributed_mode(args)

    # pdb.set_trace()

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
    vision_tower.to(device="cuda")

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()  # Model should be in evaluation mode for inference

    # Prompt model to return grounded conversations
    # instruction = 'Could you provide a detailed analysis of artifacts in this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.'
    instruction = 'Please provide a detailed analysis of artifacts in this photo, considering physical artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), structural artifacts (e.g., deformed objects, asymmetry, or distorted text), and distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation). Output with interleaved segmentation masks for the corresponding parts of the answer.'

    # Create output directory if not exists already
    os.makedirs(args.output_dir, exist_ok=True)

    # Create DDP Dataset
    img_dir = args.img_dir
    dataset = GCGEvalDDP(img_dir)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=2,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)


    output_path = os.path.join(args.output_dir,f"{img_dir.split('/')[-1]}_result.json")
    result = []

    total_metrics = {
                    'riou_score_new_max': 0, 
                    'riou_score_new_topk': 0,
                    'riou_score_with_penalty': 0,
                    'riou_score_no_penalty': 0,
                    'miou': 0, 
                    'miou_without_background': 0, 
                    'f1_score': 0,
                    'iou': 0,
                    'ap0': 0,
                    'ap05': 0,
                    'ap10': 0,
                    'ap15': 0,
                    'ap20': 0,
                    'ap25': 0,
                    'ap30': 0,
                    'ap35': 0,
                    'ap40': 0,
                    'ap45': 0,
                    'ap50': 0,
                    }
    
    # Iterate over all the images, run inference and save results
    for (image_id, image_path) in tqdm(dataloader):
        image_id, image_path = image_id[0], image_path[0]

        raw_image = cv2.imread(image_path)
        h, w = raw_image.shape[:2]

        # pdb.set_trace()

        raw_img = cv2.imread(image_path)
        if args.apply_jpeg:
            raw_img = apply_jpeg_compression(raw_img, jpeg_quality= args.jpeg_quality, output_format="ndarray")
            # pdb.set_trace()
        elif args.apply_gaussian_noise:
            raw_img = apply_gaussian_noise(raw_img, mean=0.0, std=args.gaussian_noise_std, output_format="ndarray")
        elif args.apply_gaussian_blur:
            raw_img = apply_gaussian_blur(raw_img, kernel_size=(args.gaussian_blur_size, args.gaussian_blur_size), output_format="ndarray")
        else:
            pass

        # result_caption, pred_masks, phrases = inference(instruction, image_path)  # GLaMM Inference
        result_caption, pred_masks, phrases = inference(model, instruction, raw_img.astype(np.uint8), tokenizer, clip_image_processor, transform, args)  # GLaMM Inference

        # Convert the predicted masks into RLE format
        pred_masks_tensor = pred_masks[0].cpu()
        binary_pred_masks = pred_masks_tensor > 0

        pred_mask = torch.any(binary_pred_masks, dim=0).int()
        # pdb.set_trace()
        if args.dataset == 'ours_1000':
            gt_mask = generate_mask(gt_path=args.gt_path,image_id=image_id,mask_height=h,mask_width=w)
        # 如果是rich18k
        elif args.dataset == 'rich18k_471':
            mask_path = image_path.replace('raw_imgs','heatmaps')
            gt_mask = Image.open(mask_path).convert('L')
            to_tensor = transforms.ToTensor()  # 该转换会将图像的像素值标准化到 [0, 1] 之间，并转为 tensor
            gt_mask = to_tensor(gt_mask).squeeze(0)
            gt_mask[gt_mask > 0] = 1.0
        # pdb.set_trace()
        # 如果是loki
        elif args.dataset == 'loki_112':
            gt_mask = generate_loki_mask(gt_path=args.gt_path,image_id=image_id,mask_height=h,mask_width=w)
        else:
            pdb.set_trace()
        # visualize(pred_mask=pred_mask,gt_mask=gt_mask, name=image_id) 
        riou_score_new_topk = riou_new(gt_mask=gt_mask, pred_mask=pred_mask, mode="topk")
        riou_score_new_max = riou_new(gt_mask=gt_mask, pred_mask=pred_mask, mode="max")
        riou_score_with_penalty = riou(gt_masks=gt_mask,pred_masks=pred_mask,penalty=True)
        riou_score_no_penalty = riou(gt_masks=gt_mask,pred_masks=pred_mask,penalty=False)
        miou = calculate_miou(pred_mask, gt_mask)
        f1 = calculate_f1(pred_mask, gt_mask)
        iou = calculate_iou(pred_mask, gt_mask)
        ap0 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0)
        ap05 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.05)
        ap10 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.1)
        ap15 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.15)
        ap20 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.2)
        ap25 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.25)
        ap30 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.3)
        ap35 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.35)
        ap40 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.4)
        ap45 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.45)
        ap50 = compute_map(gt_mask=gt_mask, pred_mask=pred_mask, mode='AP', iou_threshold_start=0.5)
        miou_without_background = calculate_miou_without_background(pred_mask, gt_mask)
        # pdb.set_trace()
        uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)
        rle_masks = []
        for m in uncompressed_mask_rles:
            rle_masks.append(coco_encode_rle(m))

        # Create results dictionary
        template_dict = {
            "id": image_id,
            "caption": result_caption,
            "phrases": phrases,
            "pred_masks": rle_masks,
            "miou": miou,
            "miou_without_background": miou_without_background,
            "riou_score_new_topk": riou_score_new_topk,
            "riou_score_new_max": riou_score_new_max,
            "riou_score_with_penalty": riou_score_with_penalty,
            "riou_score_no_penalty": riou_score_no_penalty,
            "f1_score": f1,
            "iou": iou,
            "ap0": ap0,
            "ap05": ap05,
            "ap10": ap10,
            "ap15": ap15, 
            "ap20": ap20,
            "ap25": ap25,
            "ap30": ap30,
            "ap35": ap35,
            "ap40": ap40,
            "ap45": ap45,
            "ap50": ap50,
        }

        result.append(template_dict)

        for metric, values in total_metrics.items():
            total_metrics[metric] += template_dict[metric]
    
    # 计算平均值
    avg_metrics = {metric: value / len(dataloader) for metric, value in total_metrics.items()}
    print(avg_metrics)
    print((avg_metrics['ap0']+avg_metrics['ap05']+avg_metrics['ap10']+avg_metrics['ap15']+avg_metrics['ap20']+avg_metrics['ap25']+avg_metrics['ap30']+avg_metrics['ap35']+avg_metrics['ap40']+avg_metrics['ap45']+avg_metrics['ap50'])/11)

    result.append(avg_metrics)

    # Save the inference results
    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=4)