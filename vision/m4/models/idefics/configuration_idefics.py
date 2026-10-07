# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" LLaMA model configuration"""
import copy
import os
from typing import Union

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class IdeficsVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.
    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        embed_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer. (elsewhere referred to as `embed_dim`)
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
    """
    model_type = "idefics"
    attribute_map = {
        "hidden_size": "embed_dim",
    }

    def __init__(
        self,
        # Case for when idefics is from the hub with no vision_model_name
        vision_model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        image_size=None,
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        vision_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config

        # idefics case (necessary for loading the idefics model)
        if hasattr(vision_config, "embed_dim"):
            self.embed_dim = vision_config.embed_dim
        # clip case (necessary for initialization)
        elif hasattr(vision_config, "hidden_size"):
            self.embed_dim = vision_config.hidden_size

        # Image size is always the size of the vision_model_name config.
        # It will be overridden if image_size is provided
        if image_size is not None and hasattr(vision_config, "image_size"):
            self.image_size = image_size
            if image_size != vision_config.image_size:
                logger.warning(
                    f"The vision model image_size {vision_config.image_size} is being overridden by the new image_size"
                    f" {self.image_size}"
                )
        elif hasattr(vision_config, "image_size"):
            self.image_size = vision_config.image_size
        else:
            raise ValueError(f"Vision config does not have an {image_size}")

        super().__init__(**kwargs)


class IdeficsPerceiverConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.
    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        use_resampler (`bool`, *optional*, defaults to `False`):
            Whether or not to use the resampler
        resampler_n_latents (`int`, *optional*, defaults to ):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 6):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        qk_layer_norms_perceiver (`bool`, *optional*, defaults to `False`):
            Whether or not to use qk layer norms in perceiver
    """
    model_type = "idefics"

    def __init__(
        self,
        use_resampler=False,
        resampler_n_latents=64,
        resampler_depth=6,
        resampler_n_heads=16,
        resampler_head_dim=96,
        qk_layer_norms_perceiver=False,
        **kwargs,
    ):
        self.use_resampler = use_resampler
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.resampler_head_dim = resampler_head_dim
        self.qk_layer_norms_perceiver = qk_layer_norms_perceiver

        super().__init__(**kwargs)


class IdeficsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.
    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        additional_vocab_size (`int`, *optional`, defaults to 0):
            Additional vocabulary size of the model, typically for the special "<img>" token. Additional vocab tokens
            are always trainable whereas regular vocab tokens can be frozen or not.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Idefics model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~IdeficsModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        alpha_initializer (`str`, *optional*, defaults to `"zeros"`):
            Initialization type for the alphas.
        alphas_initializer_range (`float`, *optional*, defaults to 0.0):
            The standard deviation of the truncated_normal_initializer for initializing the alphas in the Gated Cross
            Attention.
        alpha_type (`str`, *optional*, defaults to `"float"`):
            Whether the gating alphas should be vectors or single floats.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0)
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1)
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2)
            End of stream token id.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        cross_layer_interval (`int`, *optional*, default to 1)
            Interval for cross attention (from text to image) layers.
        qk_layer_norms (`bool`, *optional*, defaults to `False`): Whether to add layer norm after q and k
        freeze_text_layers (`bool`, *optional*, defaults to `True`): Whether to freeze text layers
        freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing text layers when `freeze_text_layers` is `True`
        freeze_lm_head (`bool`, *optional*, defaults to `False`): Whether to freeze lm head
        freeze_vision_layers (`bool`, *optional*, defaults to `True`):  Whether to freeze vision layers
        freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
        use_resampler (`bool`, *optional*, defaults to `False`): Whether to use the Resampler
        vision_config (`IdeficsVisionConfig`,  *optional*): Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig`,  *optional*): Custom perceiver config or dict
    Example:
    ```python
    >>> from transformers import IdeficsModel, IdeficsConfig
    >>> # Initializing a Idefics idefics-9b style configuration
    >>> configuration = IdeficsConfig()
    >>> # Initializing a model from the idefics-9b style configuration
    >>> model = IdeficsModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "idefics"
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        additional_vocab_size=0,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        alpha_initializer="zeros",
        alphas_initializer_range=0.0,
        alpha_type="float",
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        cross_layer_interval=1,
        qk_layer_norms=False,
        freeze_text_layers=True,
        freeze_text_module_exceptions=[],
        freeze_lm_head=False,
        freeze_vision_layers=True,
        freeze_vision_module_exceptions=[],
        use_resampler=False,
        vision_config=None,
        perceiver_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.alpha_initializer = alpha_initializer
        self.alphas_initializer_range = alphas_initializer_range
        self.alpha_type = alpha_type
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.cross_layer_interval = cross_layer_interval
        self.qk_layer_norms = qk_layer_norms
        self.freeze_vision_layers = freeze_vision_layers

        self.freeze_text_layers = freeze_text_layers
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions
        self.freeze_lm_head = freeze_lm_head

        self.use_resampler = use_resampler

        if perceiver_config is None:
            self.perceiver_config = IdeficsPerceiverConfig()
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = IdeficsPerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, IdeficsPerceiverConfig):
            self.perceiver_config = perceiver_config

        if vision_config is None:
            self.vision_config = IdeficsVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = IdeficsVisionConfig(**vision_config)
        elif isinstance(vision_config, IdeficsVisionConfig):
            self.vision_config = vision_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # IMPORTANT: Do not do any __init__ args-based checks in the constructor, since
        # PretrainedConfig.from_dict first instantiates the class with the config dict and only then
        # updates the config object with `kwargs` from from_pretrained, so during the instantiation
        # of this object many attributes have default values and haven't yet been overridden.
        # Do any required checks inside `from_pretrained` once the superclass' `from_pretrained` was run.

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        output["vision_config"] = self.vision_config.to_dict()
        output["perceiver_config"] = self.perceiver_config.to_dict()
        output["model_type"] = self.__class__.model_type

        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        outputs = super(IdeficsConfig, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        return outputs
