# Smol Models 🤏

Welcome to Smol Models, a family of efficient and lightweight AI models from Hugging Face. Our mission is to create powerful yet compact models, for text and vision, that can run effectively on-device while maintaining strong performance.

**News 📰**
- We just released SmolLM2 paper: https://arxiv.org/abs/2502.02737 

## 💬 SmolLM2 (Language Model)
[SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) is our family of compact language models available in three sizes:
- **SmolLM2-135M**: Ultra-lightweight model for basic text tasks
- **SmolLM2-360M**: Balanced model for general use
- **SmolLM2-1.7B**: Our most capable language model, available at **🤏 SmolLM2-1.7B-Instruct** [here](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

All models have instruction-tuned versions optimized for assistant-like interactions. Find them in our [SmolLM2 collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

## 👁️ SmolVLM (Vision Language Model)
[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) is our compact multimodal model that can:
- Process both images and text and perform tasks like visual QA, image description, and visual storytelling
- Handle multiple images in a single conversation
- Run efficiently on-device

## Repository Structure
```
smollm/
├── text/               # SmolLM2 related code and resources
├── vision/            # SmolVLM related code and resources
└── tools/             # Shared utilities and inference tools
    ├── smol_tools/    # Lightweight AI-powered tools
    ├── smollm_local_inference/
    └── smolvlm_local_inference/
```

## Getting Started

### SmolLM2
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

messages = [{"role": "user", "content": "Write a 100-word article on 'Benefits of Open-Source in AI research"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
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
- [SmolLM2 Documentation](text/README.md)
- [SmolVLM Documentation](vision/README.md)
- [Local Inference Guide](tools/README.md)

### Pretrained Models
- [SmolLM2 Models Collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)
- [SmolVLM Model](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

### Datasets
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) - Our instruction-tuning dataset
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) - Mathematics pretraining dataset
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - Educational content pretraining dataset
