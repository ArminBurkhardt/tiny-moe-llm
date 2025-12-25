import torch
from torch import nn
from transformers import AutoConfig, AutoModel


# LLM Encoder Module

class __Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, ffn_dim: int, num_hidden_layers: int, dropout_rate: float):
        super(__Encoder, self).__init__()
        self.config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "ffn_dim": ffn_dim,
            "num_hidden_layers": num_hidden_layers,
            "dropout_rate": dropout_rate
        }
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout_rate,
                activation='gelu'
            ) for _ in range(num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        x = self.norm(x)
        return x







class Gemma3Encoder(nn.Module):
    """Wrapper that exposes Gemma 3 1B hidden states for encoder-style use.

    Parameters
    ----------
    model_dir:
        Local path to the Gemma 3 checkpoint directory.
    target_layer:
        1-based index of the transformer layer to read from. Defaults to the
        final layer. Hidden states include the embedding output at index 0, so
        target_layer=1 returns the output after the first block.
    drop_last_n_layers:
        Alternative to target_layer; return the hidden state n layers
        before the end. Cannot be combined with target_layer.
    torch_dtype:
        Optional dtype override (defaults to bfloat16 as recommended by Gemma).
    device_map:
        Passed to AutoModel.from_pretrained to control device placement.
    """

    def __init__(
        self,
        model_dir: str,
        target_layer: int | None = None,
        drop_last_n_layers: int = 0,
        torch_dtype: torch.dtype | None = torch.bfloat16,
        device_map: str | dict | None = None,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_dir)
        num_layers = getattr(self.config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Gemma3 config missing num_hidden_layers")

        if target_layer is not None and drop_last_n_layers:
            raise ValueError("Use either target_layer or drop_last_n_layers, not both")

        if target_layer is None:
            target_layer = num_layers - drop_last_n_layers

        if target_layer < 1 or target_layer > num_layers:
            raise ValueError(
                f"target_layer must be between 1 and {num_layers}, got {target_layer}"
            )

        self.target_layer = target_layer

        self.model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            output_hidden_states=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        selected = hidden_states[self.target_layer]

        if return_all_hidden_states:
            return selected, hidden_states

        return selected


    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size



## Gemma3: https://huggingface.co/google/gemma-3-1b-it
