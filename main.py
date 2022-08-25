import os
import datetime

import torch
from torch import autocast
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline


load_dotenv()

TOKEN = os.environ.get("TOKEN")

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=TOKEN)
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=TOKEN)
pipe.to(device)

prompt = "a photograph of an astronaut riding a horse"

# 公式ドキュメントに記載がないが、pipe(prompt) の前に with autocast("cuda"): を入れておかないと float16 と float32 が競合してエラーになる
# refs: https://td2sk.hatenablog.com/entry/2022/08/24/001630
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]
# you can save the image with
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(f"dist/astronaut_rides_horse_{now}.png")
