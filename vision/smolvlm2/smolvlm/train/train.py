import os
import math
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import wandb

import torch
import transformers
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoProcessor,
    TrainingArguments,
    set_seed
)


import torch.distributed as dist
# LoRA / PEFT if needed
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

# BitsAndBytes if needed
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from smolvlm.train.smolvlm_trainer import SmolVLMTrainer
from smolvlm.train.args import DataArguments, ModelArguments, TrainingArguments
from smolvlm.datasets.builder import make_supervised_data_module

logger = logging.getLogger(__name__)

#TODO: check what these do. 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def trainer_save_model_safe(trainer: SmolVLMTrainer):
    """
    Safely saves the model if not in DeepSpeed ZeRO stage-3.
    """
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(trainer.args.output_dir, state_dict=cpu_state_dict)


def get_nb_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Returns (trainable_params, total_params) across the entire model.
    """
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, total_params


def set_trainable_params(model: torch.nn.Module, training_args: TrainingArguments):
    """
    Freezes all parameters, then selectively unfreezes based on user flags:
     - tune_vision_tower => unfreeze vision tower
     - tune_mm_connector => unfreeze connector
     - tune_language_model => unfreeze base language model
    Prints out which modules are unfrozen/frozen for clarity.
    """

    
    for param_name, param in model.named_parameters():
        param.requires_grad = False

    vis_unfrozen, conn_unfrozen, llm_unfrozen = 0, 0, 0
    vis_total, conn_total, llm_total = 0, 0, 0

    for name, param in model.named_parameters():
        if ("vision_model" in name) or ("vision_tower" in name):
            vis_total += param.numel()
            if training_args.tune_vision_tower:
                param.requires_grad = True
                vis_unfrozen += param.numel()

        elif ("connector" in name) or ("modality_projection" in name) or ("merger" in name):
            conn_total += param.numel()
            if training_args.tune_mm_connector:
                param.requires_grad = True
                conn_unfrozen += param.numel()
        else:
            
            llm_total += param.numel()
            if training_args.tune_language_model:
                param.requires_grad = True
                llm_unfrozen += param.numel()

    # Summaries
    trainable_params, total_params = get_nb_trainable_parameters(model)
    pct = 100.0 * trainable_params / max(total_params, 1)

    logger.info("=== Freeze/Unfreeze Summary ===")
    logger.info(f"  tune_vision_tower={training_args.tune_vision_tower} "
                f"(unfrozen {vis_unfrozen:,d}/{vis_total:,d} params)")
    logger.info(f"  tune_mm_connector={training_args.tune_mm_connector} "
                f"(unfrozen {conn_unfrozen:,d}/{conn_total:,d} params)")
    logger.info(f"  tune_language_model={training_args.tune_language_model} "
                f"(unfrozen {llm_unfrozen:,d}/{llm_total:,d} params)")
    logger.info(f"  => Overall trainable params: {trainable_params:,d} / {total_params:,d} "
                f"({pct:.2f}%)\n")


def enable_gradient_checkpointing(model: torch.nn.Module, training_args: TrainingArguments):
    """
    Enables gradient checkpointing if specified in TrainingArguments.
    If the model's LLM submodule supports enabling input grads, we do so (like Qwen does).
    Otherwise, we attach a forward hook to the input embedding to require grads.
    """

    logger.info("Enabling gradient checkpointing in the model.")
    model.config.use_cache = False
    model.config.use_reentrant_checkpointing = False
    # model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "model") and hasattr(model.model, "llm"):
        # Some models have a method like `enable_input_require_grads`.
        if hasattr(model.model.llm, "enable_input_require_grads"):
            logger.info("Calling model.model.llm.enable_input_require_grads() for better GC.")
            model.model.llm.enable_input_require_grads()
        else:
            # fallback approach
            logger.info("Attaching a forward hook to require grad on embeddings output.")
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            input_embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
            if input_embed is not None:
                input_embed.register_forward_hook(make_inputs_require_grad)


def prepare_model(
    model_args: ModelArguments,
    training_args: TrainingArguments
):
    """
    Loads and configures the smolVLM model (Idefics3ForConditionalGeneration),
    applying rope scaling if needed, plus optional bitsandbytes quant config.
    """
    logger.info("Loading config from %s", model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    # Possibly adjust rope scaling for a longer context
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        factor = math.ceil(training_args.model_max_length / orig_ctx_len)
        config.rope_scaling = {"type": "linear", "factor": factor}
        logger.info(f"Auto rope scaling => from {orig_ctx_len} to {training_args.model_max_length}. Factor={factor}")

    # For training, disable cache to reduce memory usage
    config.use_cache = False

    bnb_args = {}
    # If using bitsandbytes in 4- or 8-bit
    if BitsAndBytesConfig is not None and training_args.bits in [4, 8]:
        bnb_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(training_args.bits == 4),
            load_in_8bit=(training_args.bits == 8),
            llm_int8_skip_modules=["lm_head"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,
        )
        logger.info(f"Using bitsandbytes quantization: bits={training_args.bits}, type={training_args.quant_type}")

    # Possibly set attention impl
    # if training_args.disable_flash_attn2:
    #     attn_impl = "sdpa"
    # else: 
    #     attn_impl = "flash_attention_2"
    # logger.info("Instantiating SmolVLMForConditionalGeneration with attention impl=%s", attn_impl)

    if training_args.disable_flash_attn2:
        config._attn_implementation = "sdpa"
    else: 
        config._attn_implementation = "flash_attention_2"

    if model_args.frames_per_clip > 1:
        from smolvlm.model.modeling_smollmm import SmolLMMForConditionalGeneration
        config.frames_per_clip = model_args.frames_per_clip
        model_cls = SmolLMMForConditionalGeneration
        logger.info(f"Using frame emmbedding averaging of {model_args.frames_per_clip} frames")
    else:
        from smolvlm.model.modeling_smolvlm import SmolVLMForConditionalGeneration
        model_cls = SmolVLMForConditionalGeneration
    
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        config = config,
        **bnb_args,
    )

    return model


def apply_peft(model: torch.nn.Module, training_args: TrainingArguments) -> torch.nn.Module:
    """
    Applies LoRA/PEFT if training_args.peft_enable is True.
    Also calls `prepare_model_for_kbit_training` if bits=4 or 8.
    """

    if (LoraConfig is None) or (get_peft_model is None):
        raise ValueError("PEFT is not installed, but peft_enable=True was set.")

    logger.info("PEFT/LoRA is enabled. Building LoRA config...")

    # If user hasn't provided specific modules, pick a guess
    peft_target_modules = training_args.target_modules
    if not peft_target_modules:
        peft_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # For 4bit/8bit, we can do some prep:
    if training_args.bits in [4, 8] and prepare_model_for_kbit_training is not None:
        logger.info("Running `prepare_model_for_kbit_training` for LoRA + 4/8-bit support.")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    lora_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=peft_target_modules,
        bias=training_args.lora_bias,  # "none"/"all"/"lora_only"
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied. Trainable parameters:")
    model.print_trainable_parameters()
    return model


def auto_resume_or_start(training_args: TrainingArguments) -> bool:
    """
    Detect if there's a previous checkpoint to resume from.
    Return True if we found a checkpoint. Otherwise, we start fresh.
    """
    ckpts = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    return len(ckpts) > 0


def train():
    """
    Main fine-tuning entry point for your smolVLM model,
    with optional LoRA + bitsandbytes, and prints which submodules are frozen/unfrozen.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Parsing arguments with HfArgumentParser...")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if data_args.packed and model_args.apply_diagonal_block_attention:
        from smolvlm.model.varlen_packing import apply_varlen_patch
        apply_varlen_patch()
    elif data_args.packed and not model_args.apply_diagonal_block_attention:
        logger.warn("Sequence packing has being enabled WITHOUGH diagonal block attention!")
    elif not data_args.packed and model_args.apply_diagonal_block_attention:
        logger.warn("diagonal block attention has been enabled WITHOUT sequence packing. Ignoring flag!")    
        
    # Ensure reproducibility
    set_seed(training_args.seed)
    
    # Initialize wandb only on the main process (global rank 0) if wandb logging is enabled
    if "wandb" in training_args.report_to and training_args.local_rank == 0 and dist.get_rank() == 0:
        os.environ["WANDB_PROJECT"] = "smolvlmvideo"  # Set project name
        wandb.init(
            name=training_args.run_name,
            config=training_args.to_dict(),
        )
        # Ensure other processes will not try to log
        os.environ["WANDB_MODE"] = "offline"


    # Possibly set tune flags automatically based on user-provided LR
    training_args.tune_language_model = training_args.language_model_lr > 1e-9
    training_args.tune_mm_connector = training_args.connector_lr > 1e-9
    training_args.tune_vision_tower = training_args.vision_tower_lr > 1e-9

    # 1) Prepare model + config
    logger.info("Preparing model + config (possibly with bitsandbytes) ...")
    model = prepare_model(model_args, training_args)

    # 2) Freeze/unfreeze based on user flags, plus prints
    set_trainable_params(model, training_args)

    # 3) Possibly enable gradient checkpointing
    if training_args.gradient_checkpointing:
        enable_gradient_checkpointing(model, training_args)

    # 4) Possibly apply LoRA/PEFT
    if training_args.peft_enable:
        model = apply_peft(model, training_args)

    # 5) Load processor (tokenizer + image processor, etc.)
    #import ipdb; ipdb.set_trace()
    logger.info("Loading AutoProcessor from %s", model_args.model_name_or_path)
    if model_args.frames_per_clip > 1:
        from smolvlm.model.processing_smollmm import SmolLMMProcessor
        processor = SmolLMMProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side=model_args.padding_side,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side=model_args.padding_side,
            trust_remote_code=model_args.trust_remote_code,
        )

    # 6) Build dataset + collator
    logger.info("Building dataset + collator...")
    data_module = make_supervised_data_module(processor, data_args, training_args, model_args)

    # 7) Initialize custom trainer
    logger.info("Initializing SmolVLMTrainer...")
    trainer = SmolVLMTrainer(
        model=model,
        args=training_args,
        **data_module
    )

    # 8) Possibly auto-resume from checkpoint
    resume_training = auto_resume_or_start(training_args)
    if resume_training:
        logger.info("Resuming from a previous checkpoint in %s ...", training_args.output_dir)
        trainer.train(resume_from_checkpoint=True)
    else:
        logger.info("Starting a fresh training run...")
        trainer.train()

    # 9) Post-training final steps
    logger.info("Training completed. Saving final model...")
    # Re-enable model cache if needed
    model.config.use_cache = True

    # Save trainer state
    trainer.save_state()

    # Save final model (special logic if in Deepspeed)
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)

    logger.info("All done. Exiting successfully.")


if __name__ == "__main__":
    train()