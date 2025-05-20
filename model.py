from diffusers import StableDiffusionPipeline
import torch

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "ckpts/logo-mini-v1", torch_dtype=torch.float16
    ).to("cuda")
    return pipe

pipe = load_pipeline()
