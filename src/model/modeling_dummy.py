from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoModel, AutoConfig
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput


@dataclass
class DummyModelOutput(ModelOutput):
    hidden_states: Tuple[torch.Tensor]


class DummyModelConfig(PretrainedConfig):
    model_type = "dummy_model"

    def __init__(self, hidden_size: int = 768, **kwargs) -> None:
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
        return


class DummyModel(PreTrainedModel):
    config_class = DummyModelConfig

    def __init__(self, config: DummyModelConfig) -> None:
        super().__init__(config)
        self.param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        return

    def forward(self, x: torch.Tensor, **kwargs) -> DummyModelOutput:
        assert x.shape[-1] == self.config.hidden_size
        dtype = self.param.dtype
        x = x.to(dtype)
        return DummyModelOutput(hidden_states=(x,))


AutoConfig.register("dummy_model", DummyModelConfig)
AutoModel.register(DummyModelConfig, DummyModel)

if __name__ == "__main__":
    config = DummyModelConfig(hidden_size=3)
    print(config)
    model = AutoModel.from_config(config)
    print(model)
    x = torch.randn(1, 2, 3)
    output = model(x)
    print(output)
