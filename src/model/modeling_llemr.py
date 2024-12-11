from typing import Optional, List

import torch
from transformers import LlavaForConditionalGeneration


class LlemrForConditionalGeneration(LlavaForConditionalGeneration):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_values_is_padding: torch.BoolTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if pixel_values is not None and pixel_values_is_padding is not None:
            pixel_values = pixel_values[~pixel_values_is_padding].unsqueeze(1)
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_image_patches = image_features.shape[1]
        assert num_image_patches == 1, "Only one image patch is supported."
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        assert left_padding, "Input ids should be left-padded."
        (
            final_embedding,
            final_attention_mask,
            final_labels,
            position_ids
        ) = super()._merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return final_embedding, final_attention_mask, final_labels, position_ids

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None,
        pixel_values_is_padding=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        model_inputs["pixel_values_is_padding"] = pixel_values_is_padding
        return model_inputs
