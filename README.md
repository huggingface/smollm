# Smol Models 🤏

Welcome to Smol Models, a family of efficient and lightweight AI models from Hugging Face. Our mission is to create fully open powerful yet compact models, for text and vision, that can run effectively on-device while maintaining strong performance.

## [NEW] SmolLM3 (Language Model)
![image](https://github.com/user-attachments/assets/2bf61ea2-8d2e-426b-ba40-0242d34325d2)

Our 3B model outperforms Llama 3.2 3B and Qwen2.5 3B while staying competitive with larger 4B alternatives (Qwen3 & Gemma3). Beyond the performance numbers, we're sharing exactly how we built it using public datasets and training frameworks.

Resources:
- [SmolLM3-Base](https://hf.co/HuggingFaceTB/SmolLM3-3B-Base)
- [SmolLM3](https://hf.co/HuggingFaceTB/SmolLM3-3B)
- [blog](https://hf.co/blog/smollm3)

Summary:
- **3B model** trained on 11T tokens, SoTA at the 3B scale and competitive with 4B models
- **Fully open model**, open weights + full training details including public data mixture and training configs
- **Instruct model** with **dual mode reasoning,** supporting think/no_think modes
- **Multilingual support** for 6 languages: English, French, Spanish, German, Italian, and Portuguese
- **Long context** up to 128k with NoPE and using YaRN

![image](https://github.com/user-attachments/assets/f1b76d3b-af2b-4218-91b3-4ce815bdf0a8)

## 👁️ SmolVLM (Vision Language Model)
[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) is our compact multimodal model that can:
- Process both images and text and perform tasks like visual QA, image description, and visual storytelling
- Handle multiple images in a single conversation
- Run efficiently on-device

## Repository Structure
```
smollm/
├── text/               # SmolLM3/2/1 related code and resources
├── vision/            # SmolVLM related code and resources
└── tools/             # Shared utilities and inference tools
    ├── smol_tools/    # Lightweight AI-powered tools
    ├── smollm_local_inference/
    └── smolvlm_local_inference/
```

## Getting Started

### SmolLM3
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

# prepare the model input
prompt = "Give me a brief explanation of gravity in simple terms."
messages_think = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate the output
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

# Get and decode the output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

### SmolVLM
```python
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What's in this image?"}
        ]
    }
]
```

## Ecosystem
<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/RvHjdlRT5gGQt5mJuhXH9.png" width="700"/>
</div>

## Resources

### Documentation
- [SmolLM3 Documentation](text/README.md)
- [SmolLM2 paper](https://arxiv.org/abs/2502.02737v1)
- [SmolVLM Documentation](vision/README.md)
- [Local Inference Guide](tools/README.md)

### Pretrained Models
- [SmolLM3 Models Collection](https://huggingface.co/collections/HuggingFaceTB/smollm3-686d33c1fdffe8e635317e23)
- [SmolLM2 Models Collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)
- [SmolVLM Model](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

### Datasets
- [SmolLM3 Pretraining dataset](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets-685a7353fdc01aecde51b1d9)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) - Our instruction-tuning dataset
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) - Mathematics pretraining dataset
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - Educational content pretraining dataset
