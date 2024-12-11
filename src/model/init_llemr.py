import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaConfig

from src.model.modeling_dummy import DummyModelConfig
from src.model.modeling_llemr import LlemrForConditionalGeneration
from src.utils import project_path


def init_llemr(llm_pretrained_model_name_or_path, hidden_size=768):
    # dummy vision model
    vision_config = DummyModelConfig(hidden_size=hidden_size)

    # llm model
    llm_model = AutoModelForCausalLM.from_pretrained(llm_pretrained_model_name_or_path)
    llm_config = llm_model.config

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_pretrained_model_name_or_path, padding_side="left")
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    if tokenizer.pad_token_id is None:
        logging.warning("Set pad_token to unk_token")
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # llemr config
    llemr_config = LlavaConfig(vision_config, llm_config)
    llemr_config.vision_feature_layer = -1
    llemr_config.vision_feature_select_strategy = "full"
    llemr_config.pad_token_id = tokenizer.pad_token_id
    llemr_config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")

    # llemr
    llemr = LlemrForConditionalGeneration(llemr_config)
    llemr.language_model = llm_model
    llemr.resize_token_embeddings(len(tokenizer))

    # template
    if llm_pretrained_model_name_or_path == "lmsys/vicuna-7b-v1.5":
        logging.warning("Load local chat template `./chat_templates/vicuna.jinja` for Vicuna")
        with open(os.path.join(project_path, "chat_templates/vicuna.jinja")) as f:
            chat_template = f.read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        tokenizer.chat_template = chat_template

    return llemr, tokenizer


if __name__ == "__main__":
    import torch

    # llm_pretrained_model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
    llm_pretrained_model_name_or_path = "lmsys/vicuna-7b-v1.5"
    device = "cuda:0"
    llemr, tokenizer = init_llemr(llm_pretrained_model_name_or_path, hidden_size=1027)
    llemr.to(device)
    print(llemr)
    # print(tokenizer)
    prompts = ["<image><image>\nWhat is shown in the given image?", "<image>\nHello?"]
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
    pixel_values = torch.randn(2, 2, 1027).to(device)
    pixel_values_is_padding = torch.tensor([[False, False], [False, True]]).to(device)
    outputs = llemr.generate(
        **inputs,
        pixel_values=pixel_values,
        pixel_values_is_padding=pixel_values_is_padding,
        max_new_tokens=20
    )
    print(tokenizer.decode(outputs[0]))
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(tokenizer.decode(outputs[1]))
    print(tokenizer.decode(outputs[1], skip_special_tokens=True))
