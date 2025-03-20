import os
import cv2
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pycocotools import mask
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import GCG_QUESTIONS
import pdb

class GCGBaseDataset(torch.utils.data.Dataset):
    """
    Dataset Class for Grounded Conversation Generation (GCG) proposed in GLaMM.
    """
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True,
                 image_dir='', json_path=''):
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.validation = validation
        self.random_sampling = random_sampling

        self.question_templates = GCG_QUESTIONS
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation

        # Defining paths
        self.base_dir = dataset_dir
        self.image_folder = os.path.join(image_dir)
        self.ann_file = os.path.join(self.base_dir, "train", json_path)
        with open(self.ann_file, "r") as file:
            datas = json.load(file)
        self.epoch_samples = len(datas)
        self.data_infos = self._load_annotations(self.ann_file)

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['file_name'])
        annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                       'file_name': ann_info['file_name']}
        width, height = Image.open(image_path).size
        annotations['caption'] = ann_info['caption'].strip('"').strip()

        for word, grounding in ann_info["groundings"].items():
            annotations['labels'].append(word)
            annotations['tokens_positive'].append(grounding["token_positives"])

            # Convert segmentation to binary mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for rle in grounding["rle_masks"]:
                m = mask.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()
            annotations['masks'].append(binary_mask)

        return annotations

    def __getitem__(self, index):
        while True:
            ann_info = self.data_infos[index] if (self.validation or not self.random_sampling) \
                else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            if len(ann['labels']) > 0:
                break
            else:
                index = random.randint(0, len(self.data_infos) - 1)
        data_item = {"image_path": image_path, "filename": ann['file_name'], "caption": ann['caption'],
            "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}
        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, caption, tokens_positive):
        question = random.choice(self.question_templates).strip()

        # Prepare caption with tags
        def tag_caption(caption, tokens):
            for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
                caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
            return caption

        detailed_answer = tag_caption(caption, tokens_positive)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], detailed_answer)
        conversations.append(conv.get_prompt())
        questions = [question]
        return questions, conversations

    def process_data(self, data_item):
        data_labels = data_item['labels']
        masks = data_item['masks']
        caption = data_item['caption']
        tokens_positive = data_item['tokens_positive']
        image_path = data_item['image_path']

        # Function to sort elements based on the start index of each phrase
        def sort_by_start_index(items, order):
            return [items[i] for i in order]

        # Sort phrases based on their appearance in the sentence
        phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
        masks = sort_by_start_index(masks, phrase_order)
        data_labels = sort_by_start_index(data_labels, phrase_order)
        tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        bboxes = None

        questions, conversations = self.create_conversations(caption, tokens_positive)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1:], dtype=torch.long) * self.IGNORE_LABEL
        selected_labels = data_labels

        return (
        image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize, questions,
        selected_labels)


class LegionGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=512, num_classes_per_sample=3, validation=False, random_sampling=True):
        json_files = {'validation': "test.json", 'training': "train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = ''
        mode = "Val" if validation else "Train"
        epoch_samples = epoch_samples
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            validation, random_sampling, image_dir, json_path, )
        print('\033[92m' + "----GCG-{}: LEGION Dataset initialized----".format(mode) + '\033[0m')

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['img_file_name'])
        annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                       'file_name': ann_info['img_file_name']}
        width, height = Image.open(image_path).size
        orig_caption = ann_info['caption'].strip('"').strip()
        annotations['caption'] = orig_caption
        for detail in ann_info['refs']:
            phrase = detail['sentence']
            if phrase in annotations['caption']:
                annotations['labels'].append(phrase)
                index = annotations['caption'].find(phrase)
                end_index = index + len(phrase) if index != -1 else -1
                annotations['tokens_positive'].append([index, end_index])

                # Convert segmentation to binary mask
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in detail["segmentation"]:
                    seg = np.array(seg)
                    rles = mask.frPyObjects([seg], height, width)
                    m = mask.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                annotations['masks'].append(binary_mask)
        # Sort tokens_positive and corresponding lists
        tokens_positive = annotations['tokens_positive']
        sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
        annotations['tokens_positive'] = [tokens_positive[i] for i in sorted_indices]
        annotations['masks'] = [annotations['masks'][i] for i in sorted_indices]
        annotations['labels'] = [annotations['labels'][i] for i in sorted_indices]

        return annotations
    

    def __getitem__(self, index):
        while True:
            ann_dict = self.data_infos[index] if (self.validation or not self.random_sampling) \
                else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
            ann_info = next(iter(ann_dict.values()))
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            # Check if len(gt_phrases) > 0 and if True, break the loop
            if len(ann['labels']) > 0:
                break
            else:
                index = random.randint(0, len(self.data_infos) - 1)
        data_item = {"image_path": image_path, "filename": ann['file_name'], "caption": ann['caption'],
                     "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}

        return self.process_data(data_item)