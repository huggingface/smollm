# SmolLM3-3B evaluation scripts

We use the [LightEval](https://github.com/huggingface/lighteval/) library to benchmark our models.

## Setup

Use conda/uv/venv with `python>=3.11`.

For reproducibility, we recommend fixed versions of the libraries:

```sh
pip install uv
uv venv smol3_venv --python 3.11 
source smol3_venv/bin/activate

GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements.txt
```

## Running the evaluations

All commands below were run on 2 x H100s with 80GB of memory each, using the `vllm` backend.

### SmolLM3-3B base model

```bash
MODEL_ARGS="model_name=HuggingFaceTB/SmolLM3-3B-Base,dtype=bfloat16,max_model_length=32768,max_num_batched_tokens=32768,generation_parameters={temperature:0},tensor_parallel_size=2,gpu_memory_utilization=0.7"
lighteval vllm \
    "$MODEL_ARGS" \
    "smollm3_base.txt" \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```

The [SmolLM3 blog](https://huggingface.co/blog/smollm3) reports HumanEval+ and MBPP+ for the base-model win-rate comparison. LightEval does not ship those tasks, so they are defined as custom tasks in `tasks.py` (`custom|humaneval_plus` and `custom|mbpp_plus`) and included in `smollm3_base.txt`.

They use completion-style prompts and greedy `pass@1` against the extra EvalPlus tests (`evalplus/humanevalplus`, `evalplus/mbppplus`). Code is executed locally in a subprocess; only run this with the official eval datasets.

To reproduce the same suite with the EvalPlus CLI instead of LightEval:

```sh
pip install "evalplus[vllm]"
evalplus.evaluate --model HuggingFaceTB/SmolLM3-3B-Base --dataset humaneval --backend vllm --greedy
evalplus.evaluate --model HuggingFaceTB/SmolLM3-3B-Base --dataset mbpp --backend vllm --greedy
```

### SmolLM3-3B mid-trained model

This is a pure reasoning model, so no hybrid thinking:

```sh 
MODEL_ARGS="model_name=HuggingFaceTB/SmolLM3-3B-checkpoints,revision=it-mid-training,dtype=bfloat16,tensor_parallel_size=2,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
lighteval vllm "$MODEL_ARGS" "smollm3_instruct.txt" \
    --use-chat-template \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```

### SmolLM3-3B post-trained model

```sh
# Use /think or /no_think to enable or disable extended thinking
SYSTEM_PROMPT="/no_think" 
MODEL_ARGS="model_name=HuggingFaceTB/SmolLM3-3B,dtype=bfloat16,tensor_parallel_size=2,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
lighteval vllm "$MODEL_ARGS" "smollm3_instruct.txt" \
    --use-chat-template \
    --system-prompt "$SYSTEM_PROMPT" \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```

> [!NOTE]
> BFCL is not yet supported by LightEval, so we used a [fork](https://github.com/huggingface/gorilla/tree/smollm3/berkeley-function-call-leaderboard) of the public repo, with a dedicated [parser](https://github.com/huggingface/gorilla/blob/smollm3/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/smollm3.py) for SmolLM3-3B.