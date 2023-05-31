# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LLaMA-Adapter model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

LLAMA_ADAPTER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Cxxs/LLaMA-Adapter": "https://huggingface.co/Cxxs/LLaMA-Adapter/resolve/main/config.json", # todo fix this
}

class LlamaAdapterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaAdapterModel`]. It is used to instantiate an LLaMA-Adapter
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-Adapter-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA-Adapter model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaAdapterModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:

    ```python
    >>> from transformers import LlamaAdapterModel, LlamaAdapterConfig

    >>> # Initializing a LLaMA-Adapter llama_adapter-7b style configuration
    >>> configuration = LlamaAdapterConfig()

    >>> # Initializing a model from the llama_adapter-7b style configuration
    >>> model = LlamaAdapterModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama-adapter"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        num_prefix_layers=31,
        num_prefix_tokens=10,
        add_bias=False,
        add_scale=False,
        lora_rank=0,
        clip_dim=768,
        v_embed_dim=768,
        v_depth=8,
        v_num_heads=16,
        v_mlp_ratio=4.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.num_prefix_layers = num_prefix_layers
        self.num_prefix_tokens = num_prefix_tokens
        self.add_bais=add_bias
        self.add_scale=add_scale
        self.lora_rank=lora_rank
        # multi-modal
        self.clip_dim = clip_dim
        self.v_embed_dim = v_embed_dim
        self.v_depth = v_depth
        self.v_num_heads=v_num_heads,
        self.v_mlp_ratio=v_mlp_ratio,


        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )