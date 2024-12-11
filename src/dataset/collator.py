import logging
import warnings
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer


class InstructionTuningCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sys_prompt: str = "You are an AI assistant specialized in analyzing ICU patient data.",
        ignore_index: int = -100
    ) -> None:
        self.tokenizer = tokenizer
        self.sys_prompt = sys_prompt
        self.ignore_index = ignore_index
        self.response_template, self.response_token_ids = self.infer_response_template()

    def infer_response_template(self):
        logging.warning("Infer response template with v2")
        response_template, response_token_ids = self.infer_response_template_v2()
        if response_template == "":
            logging.warning("Infer response template with v1")
            response_template, response_token_ids = self.infer_response_template_v1()
        return response_template, response_token_ids

    def infer_response_template_v1(self) -> (str, List[int]):
        token = "Hi?"
        chat = [
            {"role": "user", "content": token},
        ]
        formatted_chat = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        response_template = formatted_chat[formatted_chat.find(token) + len(token):]
        response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
        logging.warning(f"Inferred response template: {repr(response_template)}")
        logging.warning(f"Inferred response template token ids: {response_token_ids}")
        return response_template, response_token_ids

    def infer_response_template_v2(self) -> (str, List[int]):
        token = "Hi?"
        chat = [
            {"role": "user", "content": token},
        ]
        formatted_chat_wo_gen = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_chat = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_chat_wo_gen = self.tokenizer.encode(formatted_chat_wo_gen, add_special_tokens=False)
        formatted_chat = self.tokenizer.encode(formatted_chat, add_special_tokens=False)
        response_token_ids = formatted_chat[len(formatted_chat_wo_gen):]
        response_template = self.tokenizer.decode(response_token_ids)
        logging.warning(f"Inferred response template: {repr(response_template)}")
        logging.warning(f"Inferred response template token ids: {response_token_ids}")
        return response_template, response_token_ids

    def apply_chat_template(self, q_text: str, a_text: str):
        chat = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": q_text},
            {"role": "assistant", "content": a_text}
        ]
        formatted_chat = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )
        return formatted_chat

    @staticmethod
    def pad_tensors(tensor_list, padding_value=0):
        max_num_events = max(tensor.shape[0] for tensor in tensor_list)
        feature_dim = tensor_list[0].shape[1]
        batch_size = len(tensor_list)

        padded_tensor = torch.full((batch_size, max_num_events, feature_dim), padding_value, dtype=torch.float)
        is_padding = torch.ones((batch_size, max_num_events), dtype=torch.bool)

        for i, tensor in enumerate(tensor_list):
            num_events = tensor.shape[0]
            padded_tensor[i, :num_events, :] = tensor
            is_padding[i, :num_events] = 0

        return padded_tensor, is_padding

    def mask_instruction(self, labels: torch.Tensor) -> torch.Tensor:
        for i in range(len(labels)):
            response_token_ids_start_idx = None

            for idx in torch.where(labels[i] == self.response_token_ids[0])[0]:
                if self.response_token_ids == labels[i][idx: idx + len(self.response_token_ids)].tolist():
                    response_token_ids_start_idx = idx

            if response_token_ids_start_idx is None:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the "
                    f'following instance: {self.tokenizer.decode(labels[i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                labels[i, :] = self.ignore_index
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                labels[i, :response_token_ids_end_idx] = self.ignore_index
        return labels

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        all_text = []
        all_events = []
        for data in batch:
            text = self.apply_chat_template(
                q_text=data[0],
                a_text=data[1],
            )
            all_text.append(text)
            all_events.append(data[2])

        inputs = self.tokenizer(
            all_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"]
        pixel_values, pixel_values_is_padding = self.pad_tensors(all_events)
        attention_mask = inputs["attention_mask"]
        labels = self.mask_instruction(input_ids.clone())

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values_is_padding": pixel_values_is_padding,
        }


if __name__ == "__main__":
    from src.dataset.dataset import InstructionTuningDataset
    from torch.utils.data import DataLoader
    from src.model.init_llemr import init_llemr

    # llm_pretrained_model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
    llm_pretrained_model_name_or_path = "lmsys/vicuna-7b-v1.5"
    device = "cuda:0"
    llemr, tokenizer = init_llemr(llm_pretrained_model_name_or_path, hidden_size=1027)
    llemr.to(device)

    dataset = InstructionTuningDataset(split="train", source="event")
    print(len(dataset))

    collator = InstructionTuningCollator(
        tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collator,
    )
    batch = next(iter(loader))
    print(batch["input_ids"].shape)
    print(batch["pixel_values"].shape)
    print(batch["attention_mask"].shape)
    print(batch["labels"].shape)
    print(batch["pixel_values_is_padding"].shape)

    for key, value in batch.items():
        batch[key] = value.to(device)
    with torch.no_grad():
        outputs = llemr(**batch)
    print(outputs.loss)
    print(outputs.logits.shape)

    llemr.train()
    for parameters in llemr.language_model.parameters():
        parameters.requires_grad = False
    outputs = llemr(**batch)
    print(outputs.loss)
    print(outputs.logits.shape)
    outputs.loss.backward()
    print("Success")
