# Pretraining
We use [nanotron](https://github.com/huggingface/nanotron/) library for training SmolLM, SmolLM2 and SmolLM3 base models.

SmolLM3 follows a transformer decoder architecture with tied embedding similar to SmolLM2, building on Llama architecture with some key modifications optimized for efficiency and long context performance.

**Grouped Query Attention (GQA):** We replaced multi-head attention with grouped query attention using 4 groups. Our ablations on a 3B model trained with 100B tokens from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) showed that GQA matches the performance of multi-head attention while significantly reducing the KV cache size during inference.

**NoPE:** We implemented NoPE from "[RoPE to NoRoPE and Back Again: A New Hybrid Attention Strategy](https://huggingface.co/papers/2501.18795)" (Yang et al., 2025), selectively removing rotary position embeddings from every 4th layer. This approach improves long context performance without affecting short context capabilities, as confirmed by our ablations.

**Intra-Document Masking:** During training, we use attention masking to ensure tokens from different documents in the same training sequence don't attend to each other. Similar to Llama 3, this helps with faster and more stable long context training while maintaining short context performance.

**Training Stability:** Following OLMo 2, we remove weight decay from embedding layers to improve training stability. This modification contributed to more stable training dynamics, with embedding norms naturally stabilizing at healthier values during training without impacting overall performance in our ablations.

All these changes were validated through ablations using the same 3B architecture trained on 100B tokens from FineWeb-Edu, ensuring each modification either improved performance or maintained it while offering other benefits.

Training Configuration: We use a global batch size of 2.36M tokens with 4096 sequence length, a learning rate of 2e-4, and the AdamW optimizer (beta1: 0.9, beta2: 0.95) with weight decay of 0.1 and gradient clipping of 1. We use the WSD (Warmup-Stable-Decay) scheduler, with 2000  warmup steps, and a linear decay to 0 in the final 10% training steps. We use [nanotron](https://github.com/huggingface/nanotron) framework for the training, [datatrove](https://github.com/huggingface/datatrove) for data processing and [lighteval](https://github.com/huggingface/lighteval) for evaluation. The model was trained on 384 H100 GPUs for 24 days. You can see the distributed training setup in the following figure.


## Setup

Please refer to [nanotron](https://github.com/huggingface/nanotron/) for detailed instructions on setting up your training environment and launching jobs. For SmolLM3 we use this [branch](https://github.com/huggingface/nanotron/tree/smollm3) and this [branch](https://github.com/huggingface/datatrove/tree/nouamane/avoid-s3) of [datatrove](https://github.com/huggingface/datatrove).

After setting up the environment and tokenizing the training datasets with [datatrove](https://github.com/huggingface/datatrove) (instructions available [here](https://github.com/huggingface/nanotron/blob/main/docs/nanoset.md#nanosets)), you can modify the configurations to match your number of nodes and local paths.

Below is an example of launching SmolLM1 135M training on 1 node (you can change the DP value to 8 in the config and adjust the batch size) and run:

```bash
git clone https://github.com/huggingface/nanotron
cd nanotron
# follow installation
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file smollm1/config_smollm1_135M.yaml
```

If you are working on a slurm cluster, you can modify the `launch.slurm` and launch the training with:

```bash
sbatch launch.slurm
```
> [!NOTE]
> Don't forget to create the logs directory before launching the job:

## Continual pre-training

The nanotron checkpoints for SmolLM2 models are available at: https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt. We will upload the SmolLM3 checkpoint very soon with the training logs.

You can find an example of continual pre-training in the [continual-pretraining](./continual-pretraining) folder.
