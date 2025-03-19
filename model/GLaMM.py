import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class GLaMMBaseModel:
    def __init__(self, config, **kwargs):
        super(GLaMMBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_glamm_model(self.config)

    def initialize_glamm_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self._configure_grounding_encoder(config)

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.train()


class GLaMMModel(GLaMMBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(GLaMMModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class GLaMMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = GLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.grounding_encoder.image_encoder(image.unsqueeze(0))

    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, **kwargs, ):

        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(input_ids, global_enc_images, attention_masks)
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, bboxes, input_ids, labels, attention_masks, offset
            )
        if grounding_enc_images is not None:
            # Extract grounding encoder image embeddings
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            assert image_embeddings.shape[0] == len(offset) - 1

            # Create segmentation token mask
            seg_token_mask = self._create_seg_token_mask(input_ids)

            # Process hidden states
            hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, seg_token_mask, offset)

            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, resize_list, label_list
            )

            if inference:
                return {"pred_masks": pred_masks, "gt_masks": masks_list, }
        else:
            pred_masks = None

        # Calculate losses
        return self._calculate_losses(pred_masks, masks_list, output)

    def _create_seg_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.seg_token_idx
        return torch.cat(
            [torch.zeros((mask.shape[0], 575)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, global_enc_images, bboxes, input_ids, labels, attention_masks, offset):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        bboxes_list = bboxes

        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, bboxes=bboxes_list, )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        if pred_masks:
            # Iterate over batch and compute mask-related losses
            for batch_idx, pred_mask in enumerate(pred_masks):
                if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                    gt_mask = masks_list[batch_idx]
                    # Resize gt_mask to match pred_mask if needed
                    if gt_mask.shape[0] != pred_mask.shape[0]:
                        gt_mask = gt_mask[:pred_mask.shape[0]]

                    assert gt_mask.shape[0] == pred_mask.shape[
                        0], f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"

                    # Compute Binary Cross-Entropy Loss
                    mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                    # Compute Dice Loss
                    mask_dice_loss += (
                            calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                    num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {"loss": total_loss, "ce_loss": ce_loss, "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss, }

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32,
                 bboxes=None, ):
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, bboxes=bboxes, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True, )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            # Adjusting for IMAGE_TOKEN_INDEX (assuming single image at start)
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 575), dtype=torch.bool).cuda(), seg_token_mask], dim=1, )
            # Process hidden states
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        return generated_output_ids, pred_masks
    
# @dataclass
# class LegionPredictionHeadOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     logits: Optional[torch.FloatTensor] = None
    

# @dataclass
# class LegionCausalLMOutputWithHeads(ModelOutput):
#     # lm head output
#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     past_key_values: Optional[List[torch.FloatTensor]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     image_hidden_states: Optional[torch.FloatTensor] = None
#     # predicition head outputs
#     prediction_head_outputs: Optional[LegionPredictionHeadOutput] = None
    


# class LegionForCausalLMWithHeads(GLaMMForCausalLM):
#     def __init__(self, config, num_labels: int = 2, **kwargs):
#         super().__init__(config, **kwargs)
        
#         self.num_labels = getattr(config, 'num_labels', num_labels)
#         self.prediction_head = nn.Linear(config.hidden_size, self.num_labels)
        

#         self.init_weights()
    
#     def forward(
#         self,
#         global_enc_images: torch.FloatTensor = None,
#         grounding_enc_images: torch.FloatTensor = None,
#         bboxes: torch.FloatTensor = None,
#         input_ids: torch.LongTensor = None,
#         labels: torch.LongTensor = None,
#         attention_masks: torch.LongTensor = None,
#         offset: torch.LongTensor = None,
#         masks_list: List[torch.FloatTensor] = None,
#         label_list: List[torch.Tensor] = None,
#         resize_list: List[tuple] = None,
#         inference: bool = False,
#         predict_head_labels: Optional[bool] = False,
#         prediction_head_labels: Optional[torch.LongTensor] = None,
#         **kwargs,
#     ) -> Union[Tuple, LegionCausalLMOutputWithHeads]:
        

#         outputs = super().forward(
#             global_enc_images=global_enc_images,
#             grounding_enc_images=grounding_enc_images,
#             bboxes=bboxes,
#             input_ids=input_ids,
#             labels=labels,
#             attention_masks=attention_masks,
#             offset=offset,
#             masks_list=masks_list,
#             label_list=label_list,
#             resize_list=resize_list,
#             inference=inference,
#             **kwargs
#         )
        
#         # 提取生成任务的损失
#         lm_head_loss = outputs.get('loss', None)
#         final_loss = lm_head_loss
        
#         prediction_head_outputs = None
#         if predict_head_labels:
#             if input_ids is None:
#                 raise ValueError("`input_ids` must be provided when `predict_head_labels` is True.")
#             if self.config.image_token_index is None:
#                 raise ValueError("`image_token_index` must be set in the config.")
            
#             # 提取最后一层的隐藏状态，并定位图像特征
#             hidden_states = outputs.get('hidden_states', None)
#             if hidden_states is None:
#                 raise ValueError("Hidden states are not returned by the model. Ensure `output_hidden_states=True`.")
            
#             last_layer_hidden_states = hidden_states[-1][:, :input_ids.size(-1), :]  # Shape: (batch_size, seq_len, hidden_size)
            
#             # 创建图像特征掩码
#             image_features_mask = input_ids == self.config.image_token_index  # Shape: (batch_size, seq_len)
#             shifted_image_features_mask = torch.cat([
#                 image_features_mask[:, 1:], 
#                 torch.zeros(image_features_mask.size(0), 1, dtype=torch.bool, device=image_features_mask.device)
#             ], dim=1)  # Shift right by 1, pad last with False
#             last_feature_mask = image_features_mask & ~shifted_image_features_mask  # Shape: (batch_size, seq_len)
            
#             # 提取图像特征的隐藏状态
#             multimodal_image_features = last_layer_hidden_states[last_feature_mask]  # Shape: (num_images, hidden_size)
            
#             # 通过分类头计算 logits
#             prediction_head_logits = self.prediction_head(multimodal_image_features)  # Shape: (num_images, num_labels)
            
#             prediction_head_loss = None
#             if prediction_head_labels is not None:
#                 if prediction_head_labels.size(0) != prediction_head_logits.size(0):
#                     raise ValueError("Number of `prediction_head_labels` must match number of image features.")
                
#                 # 计算分类损失
#                 loss_fct = nn.CrossEntropyLoss()
#                 prediction_head_loss = loss_fct(
#                     prediction_head_logits.view(-1, self.num_labels),
#                     prediction_head_labels.view(-1).to(prediction_head_logits.device)
#                 )
                
#                 # 将分类损失加入总损失
#                 final_loss = final_loss + prediction_head_loss if final_loss is not None else prediction_head_loss
            
#             # 封装分类头的输出
#             prediction_head_outputs = LegionPredictionHeadOutput(
#                 loss=prediction_head_loss,
#                 logits=prediction_head_logits
#             )
        
#         # 根据 `return_dict` 和 `predict_head_labels` 的值，决定返回类型
#         return LegionCausalLMOutputWithHeads(
#             loss=final_loss,
#             logits=outputs.get('output', None),
#             past_key_values=outputs.get('past_key_values', None),
#             hidden_states=outputs.get('hidden_states', None),
#             attentions=outputs.get('attentions', None),
#             image_hidden_states=outputs.get('image_hidden_states', None),
#             prediction_head_outputs=prediction_head_outputs
#         )

