import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    PreTrainedTokenizer
)

from src.dataset.collator import InstructionTuningCollator
from src.dataset.dataset import InstructionTuningDataset
from src.model.init_llemr import init_llemr
from src.model.modeling_llemr import LlemrForConditionalGeneration
from src.model.utils import find_all_linear_names

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    name_or_path: Optional[str] = field(default=None)
    llm_pretrained_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B-Instruct")
    train_type: Optional[str] = field(
        default="train_both",
        metadata={
            "help": """
            1. train_multi_modal_projector
            2. train_both
            """
        },
    )
    use_lora: Optional[bool] = field(default=True)
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    vision_hidden_size: int = 768


@dataclass
class DataArguments:
    source: Optional[str] = field(default="note")


def load_model(model_args: ModelArguments):
    if model_args.name_or_path is not None:
        logging.warning(f"Load model {model_args.name_or_path} from pretrained")
        model = LlemrForConditionalGeneration.from_pretrained(
            model_args.name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.name_or_path,
            padding_side="left"
        )
    else:
        logging.warning(f"Init model {model_args.llm_pretrained_model_name_or_path}")
        model, tokenizer = init_llemr(
            model_args.llm_pretrained_model_name_or_path, model_args.vision_hidden_size
        )

    assert model_args.train_type in ["train_multi_modal_projector", "train_both"]
    if model_args.train_type == "train_multi_modal_projector":
        logging.warning("Train multi_modal_projector")
        for param in model.language_model.parameters():
            param.requires_grad = False
    else:
        logging.warning("Train both")

    if model_args.use_lora:
        assert model_args.train_type == "train_both"
        logging.warning("Use Lora")

        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)

    else:
        logging.warning("Not use Lora")

    return model, tokenizer


def load_data(data_args: DataArguments, tokenizer: PreTrainedTokenizer):
    train_dataset = InstructionTuningDataset(
        split="train",
        source=data_args.source,
    )
    val_dataset = InstructionTuningDataset(
        split="val",
        source=data_args.source,
    )
    collator = InstructionTuningCollator(
        tokenizer=tokenizer,
    )
    return train_dataset, val_dataset, collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model(model_args)
    train_dataset, val_dataset, collator = load_data(data_args, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    tokenizer.save_pretrained(training_args.output_dir)
    trainer.train()


if __name__ == "__main__":
    train()
