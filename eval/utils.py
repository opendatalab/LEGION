import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from scipy.ndimage import label
from skimage.measure import label as label_new
import math
from scipy.optimize import linear_sum_assignment
import pdb


def grounding_image_ecoder_preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                                      pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
                                      img_size=1024) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))

    return x


def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out


def mask_to_rle_numpy(mask: np.ndarray):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    h, w = mask.shape

    # Put in fortran order and flatten h,w
    mask = np.transpose(mask).flatten()

    # Compute change indices
    diff = mask[1:] ^ mask[:-1]
    change_indices = np.where(diff)[0]

    # Encode run length
    cur_idxs = np.concatenate(
        ([0], change_indices + 1, [h * w])
    )
    btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
    counts = [] if mask[0] == 0 else [0]
    counts.extend(btw_idxs.tolist())

    return {"size": [h, w], "counts": counts}


def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    return iou


def bbox_to_x1y1x2y2(bbox):
    x1, y1, w, h = bbox
    bbox = [x1, y1, x1 + w, y1 + h]

    return bbox


####
def generate_mask(gt_path, image_id, mask_height, mask_width):
    # 读取 JSON 文件
    with open(gt_path, 'r') as file:
        data = json.load(file)
    
    # 初始化 mask，默认为背景（0）
    final_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # 使用列表推导式和短路操作来提高查找效率
    for item in data:
        for key, value in item.items():
            img_file_name = value.get("img_file_name", "")
            
            # 检查路径是否包含 image_id，若找到就停止遍历
            if image_id in img_file_name:
                # 遍历分割区域并绘制到 mask 上
                for ref in value.get("refs", []):
                    segmentation = ref.get("segmentation", [])
                    for seg in segmentation:
                        # 每个seg为[x1, y1, x2, y2, ...]，我们直接将其处理为二维坐标点
                        points = np.array(seg, dtype=np.float32).reshape((-1, 2))  # 将每个坐标对(x, y)组成一个点
                        points = points.astype(np.int32)  # 转换为整数坐标
                        cv2.fillPoly(final_mask, [points], 1)
                # pdb.set_trace()
                break  # 找到目标后退出循环


    # 将 final_mask 调整为指定的大小
    final_mask_resized = cv2.resize(final_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
    
    # 转换为 PyTorch Tensor
    mask_tensor = torch.tensor(final_mask_resized, dtype=torch.long)
    
    return mask_tensor


def generate_loki_mask(gt_path, image_id, mask_height, mask_width):
    # 读取 JSON 文件
    with open(gt_path, 'r', encoding='utf-16') as file:
        data = json.load(file)
    
    # 初始化 mask，默认全零
    final_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # 遍历 JSON 数据
    for item in data:
        img_file_name = item.get("image_path", "")  # 更新路径字段
        if image_id in img_file_name:  # 匹配图像 ID
            # 遍历标注的区域
            for region in item.get("problems", {}).get("regional", []):
                bbox = region.get("region", [])
                if len(bbox) == 4:
                    x, y, w, h = bbox  # 解析 xywh 格式
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)  # 转换为左上角 (x1, y1) 和右下角 (x2, y2)
                    
                    # 确保坐标不超出边界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(mask_width - 1, x2), min(mask_height - 1, y2)
                    
                    # 使用 OpenCV 填充矩形区域
                    cv2.rectangle(final_mask, (x1, y1), (x2, y2), 1, thickness=-1)  # thickness=-1 表示填充区域
                else:
                    raise ValueError
            
            break  # 找到目标图像后退出循环

    # 调整 mask 大小（如果需要）
    final_mask_resized = cv2.resize(final_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
    
    # 转换为 PyTorch Tensor
    mask_tensor = torch.tensor(final_mask_resized, dtype=torch.long)
    
    return mask_tensor



def find_connected_regions(binary_mask):
    """
    Find connected regions in a binary mask using scipy.ndimage.label.
    """
    # 如果输入是 PyTorch 张量，先转换为 NumPy 数组
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()
    
    # 使用 scipy.ndimage.label 找到连通区域
    labeled_mask, num = label(binary_mask)
    regions = []
    for i in range(1, num + 1):
        regions.append((labeled_mask == i))
    return regions


def riou(gt_masks, pred_masks, penalty=False):
    """
    Compute region-level score with area penalty.
    """
    # 如果输入是 PyTorch 张量，先将其转换为 NumPy 数组
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy()
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()

    # 获取连通区域
    gt_regions = find_connected_regions(gt_masks)
    pred_regions = find_connected_regions(pred_masks)

    M = len(gt_regions)  # GT 中的区域数量
    N = len(pred_regions)  # 预测中的区域数量

    # 计算总预测前景面积
    total_pred_area = pred_masks.sum()

    # 特殊情况处理
    if M == 0 and N == 0:  # 没有真值和预测，得分为 1.0
        return 1.0
    if M > 0 and N == 0:  # 有真值但没有预测，得分为 0.0
        return 0.0
    if M == 0 and N > 0:  # 有预测但没有真值，得分为 0.0
        return 0.0

    # 构建代价矩阵和 IOU 矩阵
    cost_matrix = np.zeros((M, N))
    iou_matrix = np.zeros((M, N))

    # 预计算每个预测区域的面积，便于后续使用
    pred_areas = [r.sum() for r in pred_regions]

    # 计算每个 GT 和预测区域的 IOU
    for i in range(M):
        for j in range(N):
            inter = np.logical_and(gt_regions[i], pred_regions[j]).sum()  # 交集
            union = np.logical_or(gt_regions[i], pred_regions[j]).sum()  # 并集
            iou_val = inter / union if union > 0 else 0.0
            cost_matrix[i, j] = 1 - iou_val  # 代价矩阵
            iou_matrix[i, j] = iou_val  # IOU 矩阵

    # 最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 初始化匹配的 IOU 分数和被匹配的预测区域索引集合
    matched_iou_scores = np.zeros(M)
    matched_pred_indices = set(col_ind)

    # 遍历匹配结果，保存匹配的 IOU 分数
    for i_idx, j_idx in zip(row_ind, col_ind):
        matched_iou_scores[i_idx] = iou_matrix[i_idx, j_idx]

    # 基础得分：匹配到的真值区域平均 IoU
    base_score = matched_iou_scores.mean()

    # 如果总预测前景面积为 0
    if total_pred_area == 0:
        return base_score

    # 计算匹配到的预测区域的面积之和
    matched_area = sum(pred_areas[j_idx] for j_idx in matched_pred_indices)

    # 惩罚因子：匹配到的预测区域面积 / 总预测前景面积
    penalty_factor = matched_area / total_pred_area if penalty else 1.0

    # 最终得分
    final_score = base_score * penalty_factor

    return final_score


def riou_new(gt_mask: torch.tensor, pred_mask: torch.tensor, mode: str = "topk", k: float = 0.5) -> float:
    """
    计算自定义的分割评估指标 riou。

    参数：
    - gt_mask (np.ndarray): 真值掩码，形状为 (H, W)，值为 0 或 1。
    - pred_mask (np.ndarray): 预测掩码，形状为 (H, W)，值为 0 或 1。
    - mode (str): 评估模式，值为"topk"或者"max", "max"代表取最大匹配区域的iou值, "topk"代表取前真值连通区域1/3数量的iou值计算平均.
    - k (float): IoU 阈值，默认为 0.5。

    返回：
    - riou_score[float]: 所得的riou指标值。
    """
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().numpy()
    if isinstance(pred_mask, torch.Tensor):      
        pred_mask = pred_mask.detach().numpy()
    gt_mask = gt_mask.astype(int)
    pred_mask = pred_mask.astype(int)
    # 标记连通区域，使用8连通性
    label_connected = label_new(gt_mask, connectivity=2, background=0)
    pred_connected = label_new(pred_mask, connectivity=2, background=0)
    
    M = label_connected.max()
    N = pred_connected.max()
    
    # 打印连通区域信息
    # print(f"真值连通区域数 M: {M}")
    # print(f"预测连通区域数 N: {N}")
    
    # 如果没有真值前景区域，返回空列表
    if M == 0:
        return 0
    
    # 提取每个真值前景区域的掩码
    label_regions = [label_connected == i for i in range(1, M + 1)]
    
    # 提取每个预测前景区域的掩码
    pred_regions = [pred_connected == j for j in range(1, N + 1)]
    
    iou_list = []
    
    for idx, A_i in enumerate(label_regions):
        S_i = []
        for P_j in pred_regions:
            intersection = np.logical_and(A_i, P_j).sum()
            union = np.logical_or(A_i, P_j).sum()
            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union
            if iou > k:
                S_i.append(P_j)
        
        if S_i:
            # 计算联合区域 Ui
            U_i = np.logical_or.reduce(S_i)
            intersection_Ui = np.logical_and(A_i, U_i).sum()
            union_Ui = np.logical_or(A_i, U_i).sum()
            if union_Ui == 0:
                iou_i = 0.0
            else:
                iou_i = intersection_Ui / union_Ui
        else:
            iou_i = 0.0
        
        iou_list.append(iou_i)
    
    if mode == "max":
        riou_score = max(iou_list)
    elif mode == "topk":
        topk = math.ceil(1/3 * M)
        ranked_list = sorted(iou_list,reverse=True)[:topk]
        riou_score = sum(ranked_list) / topk

    return riou_score


def calculate_iou(gt_mask, pred_mask):
    """计算IoU（交并比），返回值是两个区域之间的IoU"""
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().numpy()
    if isinstance(pred_mask, torch.Tensor):      
        pred_mask = pred_mask.detach().numpy()
    intersection = np.sum(gt_mask * pred_mask)  # 相交部分
    union = np.sum(gt_mask) + np.sum(pred_mask) - intersection  # 并集部分
    return intersection / union if union != 0 else 0.0

def calculate_ap(gt_mask, pred_mask, iou_threshold):
    """计算单个IoU阈值下的AP"""
    # 将标签转换为整数类型
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().numpy()
    if isinstance(pred_mask, torch.Tensor):      
        pred_mask = pred_mask.detach().numpy()
    gt_mask = gt_mask.astype(int)
    pred_mask = pred_mask.astype(int)

    # 标记连通区域，使用8连通性
    label_connected = label_new(gt_mask, connectivity=2, background=0)
    pred_connected = label_new(pred_mask, connectivity=2, background=0)

    M = label_connected.max()  # 真值区域的数量
    N = pred_connected.max()   # 预测区域的数量
    
    # 如果没有真值区域，返回0 AP
    if M == 0:
        return 0.0
    
    # 提取每个真值前景区域的掩码
    label_regions = [label_connected == i for i in range(1, M + 1)]
    
    # 提取每个预测前景区域的掩码
    pred_regions = [pred_connected == j for j in range(1, N + 1)]
    
    iou_matrix = np.zeros((M, N))

    # 计算IoU矩阵
    for i in range(M):
        for j in range(N):
            iou_matrix[i, j] = calculate_iou(label_regions[i], pred_regions[j])

    # 使用匈牙利算法进行区域匹配
    gt_idx, pred_idx = linear_sum_assignment(-iou_matrix)  # 找到最大IoU匹配

    # 计算AP
    tp, fp = 0, 0
    for i, j in zip(gt_idx, pred_idx):
        if iou_matrix[i, j] >= iou_threshold:
            tp += 1  # 真阳性
        else:
            fp += 1  # 假阳性
    # pdb.set_trace()
    
    return tp / (tp + fp) if (tp + fp) != 0 else 0.0  # 返回AP

def compute_map(gt_mask, pred_mask, mode='mAP', iou_threshold_start=0.5, iou_threshold_end=0.95, step=0.05):
    """
    计算mAP或AP，输入gt_mask和pred_mask是大小为H*W的tensor，mode选择'mAP'或'AP'
    :param gt_mask: 真实值掩码，numpy数组或Tensor类型
    :param pred_mask: 预测值掩码，numpy数组或Tensor类型
    :param mode: 'mAP'或'AP'，选择计算mAP或指定阈值的AP
    :param iou_threshold_start: 起始阈值
    :param iou_threshold_end: 终止阈值
    :param step: 步长
    :return: float类型的AP值（或mAP）
    """
    
    # 将Tensor转换为Numpy数组
    if isinstance(gt_mask, np.ndarray) == False:
        gt_mask = gt_mask.cpu().numpy()
    if isinstance(pred_mask, np.ndarray) == False:
        pred_mask = pred_mask.cpu().numpy()

    if gt_mask.shape != pred_mask.shape:
        pdb.set_trace()
    
    if mode == 'mAP':
        aps = []
        for iou_threshold in np.arange(iou_threshold_start, iou_threshold_end + step, step):
            ap = calculate_ap(gt_mask, pred_mask, iou_threshold)
            aps.append(ap)
        
        return np.mean(aps)  # 返回mAP

    elif mode == 'AP':
        # 计算指定iou_threshold下的AP
        return calculate_ap(gt_mask, pred_mask, iou_threshold_start)
    
    else:
        raise ValueError("Invalid mode. Choose 'mAP' or 'AP'.")
    
## disturb
import kornia
from PIL import Image

# 判断输入格式并转换为统一的torch tensor (B, C, H, W)
def convert_to_tensor(image):
    if isinstance(image, np.ndarray):  # 如果是numpy数组
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    elif isinstance(image, torch.Tensor):  # 如果是torch tensor
        if image.dim() == 3:
            image_tensor = image.unsqueeze(0).float() / 255.0
        else:
            image_tensor = image.float() / 255.0
    elif isinstance(image, Image.Image):  # 如果是PIL图像
        image = np.array(image)  # 将PIL图像转换为numpy数组
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        raise TypeError("Unsupported image type")
    return image_tensor

# 将tensor或numpy数组转换为PIL图像
def convert_to_pil(image):
    if isinstance(image, torch.Tensor):
        image = image.squeeze().permute(1, 2, 0).numpy() * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)

# 格式转换函数
def convert_format(image_tensor, output_format):
    """
    将图像转换为指定的格式：ndarray, Tensor, 或 PIL
    :param image_tensor: 输入的图像tensor（B, C, H, W）,归一化到0 ~ 1的
    :param output_format: 输出格式，支持 "ndarray", "Tensor", "PIL"
    :return: 转换后的图像
    """
    if output_format == "ndarray":
        return image_tensor.squeeze().permute(1, 2, 0).numpy() * 255.0
    elif output_format == "Tensor":
        return image_tensor.squeeze() * 255.0
    elif output_format == "PIL":
        return convert_to_pil(image_tensor)
    else:
        raise ValueError("Invalid output format specified")

# 高斯噪声函数
def apply_gaussian_noise(image, mean: float = 0.0, std: float = 0.1, output_format="ndarray"):
    """
    对图像添加高斯噪声
    :param image: 输入图像，支持np.ndarray, torch.Tensor, PIL.Image，0 ~ 255
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    :param output_format: 输出格式，支持 "ndarray", "Tensor", "PIL"
    :return: 加入噪声后的图像
    """
    image_tensor = convert_to_tensor(image)  # 转换为tensor
    mean /= 255
    # std /= 255
    noise = torch.randn_like(image_tensor) * std + mean
    image_tensor = image_tensor + noise
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)  # 确保像素值在有效范围内
    
    return convert_format(image_tensor, output_format)

# JPEG压缩函数
def apply_jpeg_compression(image, jpeg_quality: int = 50, output_format="ndarray"):
    """
    对图像施加JPEG压缩
    :param image: 输入图像，支持np.ndarray, torch.Tensor, PIL.Image，0 ~ 255
    :param jpeg_quality: JPEG压缩质量，范围 [0, 100]
    :param output_format: 输出格式，支持 "ndarray", "Tensor", "PIL"
    :return: 施加JPEG压缩后的图像
    """
    image_tensor = convert_to_tensor(image)  # 转换为tensor
    jpeg_compression = kornia.augmentation.RandomJPEG(jpeg_quality=(jpeg_quality, jpeg_quality), p=1.0)
    image_tensor = jpeg_compression(image_tensor)
    
    return convert_format(image_tensor, output_format)

# 高斯模糊函数
def apply_gaussian_blur(image, kernel_size: tuple = (3, 3), sigmaX=0, output_format="ndarray"):
    """
    对图像施加高斯模糊
    :param image: 输入图像，支持np.ndarray, torch.Tensor, PIL.Image，0 ~ 255
    :param sigma: 高斯模糊的标准差
    :param kernel_size: 卷积核大小
    :param output_format: 输出格式，支持 "ndarray", "Tensor", "PIL"
    :return: 施加高斯模糊后的图像
    """
    image_tensor = convert_to_tensor(image)  # 转换为tensor

    image_np = convert_format(image_tensor, "ndarray")
    image_np = cv2.GaussianBlur(image_np,kernel_size,sigmaX=sigmaX)

    if output_format == "ndarray":
        return image_np
    else:
        return Image.fromarray(image_np.astype(np.uint8))