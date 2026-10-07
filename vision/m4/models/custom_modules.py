import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ContextManagers

from m4.training.setup_vision_model import vision_model_name_to_model
from m4.training.utils import (
    deepspeed_zero_init_disabled_context_manager,
    is_deepspeed_zero_init_enabled,
    load_state_dict_into_model,
)


# from pathlib import Path


class VLOOMPreTrainedModelBase(PreTrainedModel):
    # The problem we are trying to solve is 2 nested zero.Init thanks to fetching from_pretrained(vision_model_name)
    # and then one more zero.Init to override from_pretrained(vision_model_name) once again as it was done in the original - this breaks deepspeed zero3 w/ zero.Init
    # So one solution is this:
    # a. replace  from_pretrained(vision_model_name) with from_config(vision_model_name) while hacking to disable zero.Init context
    # b. instead of straight replacement of model.vision_model = from_pretrained(vision_model_name) when it gets updated, we first do from_pretrained(vision_model_name) and then update the existing model with weights using the already zero.Init'ed pre-sharded weights
    #
    # there are a few variations to get_vision_model_from_config - all need to bypass zero.Init under zero3
    # 1. one variant is to hack into accelerate's deepspeed_plugin and turn off zero.Init while loading the vision model
    # 2. the other variant is to override _from_config method with our version that doesn't do zero.Init

    @classmethod
    def override_vision_model(cls, model, vision_model_name, vision_config, torch_dtype):
        vision_config["torch_dtype"] = torch_dtype
        # 1. fetch the pretrained vision model w/o zero.Init
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            vision_model = AutoModel.from_pretrained(vision_model_name, **vision_config, trust_remote_code=True)

        # this extracts the desired submodule if the part we want is nested (e.g. as in clip)
        real_vision_model = vision_model_name_to_model(vision_model_name, vision_model)

        # 2. now override the weights already sharded by zero.Init with the weights from the real_vision_model
        # by gradually gathering sharded weights and replacing with new weights
        if is_deepspeed_zero_init_enabled():
            state_dict = real_vision_model.state_dict()
            load_state_dict_into_model(model.vision_model, state_dict, start_prefix="")
        else:
            model.vision_model = real_vision_model

    @classmethod
    def from_config(cls, config, **kwargs):
        model = cls(config, **kwargs)
        return model

    @classmethod
    def from_pretrained_models(cls, *args, **kwargs):
        """
        Use this method when creating a new vloom model that hasn't been yet trained and it'll be
        composed of 2 pre-trained models - hence `pretrained_models`.
        """

        return cls.from_pretrained(*args, **kwargs, new_model=True)

    @classmethod
    def from_pretrained(cls, *model_args, is_resume=False, new_model=False, **kwargs):
        """
        Use this method when loading an already pretrained vloom model - either from a checkpoint or from hub.
        For creating an untrained model use `pretrained_models` instead.
        """
        # config is:
        # 1. either not passed and then we use the model's default config (used by tests)
        # 2. passed and in which case it's one of:
        #   2a. `PretrainedConfig` (a new m4 model)
        #   2b. path to a json config (an already pretrained m4 model, usually resumed training)
        config = kwargs.get("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(*model_args, **kwargs, return_unused_kwargs=False)
        elif not isinstance(config, PretrainedConfig):
            # adapted from https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/modeling_utils.py#L1920
            assert isinstance(config, os.PathLike)
            config_path = str(config)
            config = cls.config_class.from_pretrained(
                config_path,
                return_unused_kwargs=False,
                **kwargs,
            )

        is_untrained_vloom_model, is_pretrained_vloom_model_resumed, is_pretrained_vloom_model_from_hub_or_path = (
            False,
            False,
            False,
        )
        # 3 Cases:
        # 1 - Model has never been trained. This means we need a vision_model_name to start the training with - as we never create one from scratch -
        # 2 - Model has been trained and is resuming. We load a random empty model in this case
        # 3 - Model has been trained and saved somewhere with a path or is on the hub and has a vision_model_name we initialize the vision model from the vision_model_name class.
        if new_model:
            is_untrained_vloom_model = True
        elif is_resume:
            is_pretrained_vloom_model_resumed = True
        else:
            is_pretrained_vloom_model_from_hub_or_path = True

        # torch_dtype is crucial for using the minimal amount of memory at load time
        torch_dtype = kwargs.get("torch_dtype", None)
        vision_model_name = config.vision_config.vision_model_name

        # Create an uninitialized vision_model to insert into the main model.
        vision_model_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)
        # Override image_size if we want to increase it compared to pretraining
        if hasattr(vision_model_config, "vision_config"):
            vision_model_config.vision_config.image_size = config.vision_config.image_size
        else:
            vision_model_config.image_size = config.vision_config.image_size
        # model_with_vision_component = AutoModel.from_config(
        #     vision_model_config, torch_dtype=torch_dtype, trust_remote_code=True
        # )

        # Extracts the desired submodule if the part we want is nested (e.g. as in clip)
        # kwargs["vision_model"] = vision_model_name_to_model(vision_model_name, model_with_vision_component)

        # 1. We load a trained checkpoint but we are not resuming a training:
        # If the model is from_hub or from_path, the language model is loaded as well, and
        # the uninitialized vision_model is overridden by the checkpoint's weights (i.e. idefics' weights)```
        if is_pretrained_vloom_model_from_hub_or_path:
            model = super().from_pretrained(*model_args, **kwargs)

        # 2. We resume under deepspeed:
        # We create an empty model, and get deepspeed to load the weights from the checkpoint.
        # Not all models have these keys so handle the case they don't have them
        elif is_pretrained_vloom_model_resumed:
            _ = kwargs.pop("config", None)
            model = super().from_pretrained(None, config=config, state_dict={}, **kwargs)

        # 3. If is_untrained_vloom_model, we load the language model first, then we override
        # the uninitialized vision_model with one with pretrained weights from the model vision_model_name
        elif is_untrained_vloom_model:
            model = super().from_pretrained(*model_args, **kwargs)
            cls.override_vision_model_wrapper(
                model, config, vision_model_name, vision_model_config.to_dict(), torch_dtype
            )

        return model


class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings.
    In practise, the regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0, then it will create `num_additional_embeddings` additional parameters that are always trained.
    If `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze=False,
        device=None,
        dtype=None,
        padding_idx=None,
        **kwargs,
    ) -> None:
        """
        num_additional_embeddings: int. Number of additional embeddings. Only useful when you `partially_freeze=True`.
        partially_freeze: bool. If True, the regular `weight` will be frozen. `additional_weight` is never frozen.

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`, `max_norm` or `norm_type`. We are not supporting these.
        """
        if padding_idx is not None and padding_idx > num_embeddings:
            raise ValueError(f"padding_idx must be within num_embeddings. Got {padding_idx} and {num_embeddings}")
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=padding_idx,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        if partially_freeze:
            self.weight.requires_grad_(False)

        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings),
           since the 2nd embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do
        the padding, but then we have to create a new tensor and populate it with 2 tensors that are
        spread out across various indices - i.e. not a simple concat - I haven't benchmarked the
        complex case if it's any faster, given that seqlens are usually relatively short it's
        probably not faster or if faster not by much - but might be a good idea to measure.

        """
        if self.num_additional_embeddings == 0:
            return self.additional_embedding(input_ids)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.num_embeddings,
            self.num_additional_embeddings,
            self.embedding_dim,
            self.partially_freeze,
        )

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, **kwargs):
        raise NotImplementedError


class DecoupledLinear(nn.Linear):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters.
    In practise, the regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `out_additional_features` > 0, then it will create `out_additional_features * in_features` additional parameters that are always trained.
    If `out_additional_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_additional_features: int = 0,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        out_additional_features: int. Number of additional trainable dimensions. Only makes sense when `partially_freeze=True`.
        partially_freeze: bool. If True, the regular `weight` will be frozen and extra parameters (if any) will be trainable. If False, default to the regular behavior of nn.Linear.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.out_additional_features = out_additional_features
        self.partially_freeze = partially_freeze

        self.in_features = in_features
        self.out_features = out_features

        if partially_freeze:
            self.weight.requires_grad_(False)
            if bias:
                self.bias.requires_grad_(False)

        if out_additional_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=out_additional_features,
                bias=bias,
                device=device,
                dtype=dtype,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)

        if self.out_additional_features > 0:
            additional_features = self.additional_fc(input)
            output = torch.cat((output, additional_features), -1)

        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, out_additional_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.out_features,
            self.out_additional_features,
            self.bias is not None,
            self.partially_freeze,
        )


if __name__ == "__main__":
    emb = DecoupledEmbedding(num_embeddings=10, num_additional_embeddings=3, embedding_dim=5, partially_freeze=True)
    for n, p in emb.named_parameters():
        print(n, p.requires_grad)
    idx = torch.tensor([[11, 1, 3]])
    y = emb(idx)
    loss = y.sum()
    loss.backward()
    print(emb.weight, emb.weight.grad)
    print(emb.additional_embedding, emb.additional_embedding.grad)

    lin = DecoupledLinear(in_features=3, out_features=4, out_additional_features=2, bias=True, partially_freeze=True)
    for n, p in lin.named_parameters():
        print(n, p.requires_grad)
    x = torch.randn(12, 3)
    y = lin(x)
    loss = y.sum()
    loss.backward()
    print("Weight w and grad:", lin.weight, lin.weight.grad)
    print("bias w and grad:", lin.bias, lin.bias.grad)
    print("additional_fc.weight w and grad:", lin.additional_fc.weight, lin.additional_fc.weight.grad)
    print("additional_bias w and grad:", lin.additional_fc.bias, lin.additional_fc.bias.grad)
