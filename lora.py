import torch

# Load LoRA weights
lora_path = "./LoRA/Fant5yP0ny.safetensors"
lora_weights = load_safetensors(lora_path)


def apply_lora_weights(model, lora_weights, alpha=1.0):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_weights:
                param.add_(lora_weights[name] * alpha)


apply_lora_weights(pipeline.unet, lora_weights, alpha=0.8)
