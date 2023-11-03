import os
from typing import Optional
from io import BytesIO

import runpod
from runpod.serverless.utils.rp_validator import validate

from PIL import Image
import torch
import requests
import cv2
import numpy as np
from diffusers import (
    AutoPipelineForInpainting,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from schema import INPUT_SCHEMA
from constants import MODEL_NAME, MODEL_VARIANT


torch.cuda.empty_cache()


def load_pipeline():
    pipeline = AutoPipelineForInpainting.from_pretrained(
        MODEL_NAME,
        variant=MODEL_VARIANT,
        torch_dtype=torch.float16,
        use_safetensors=True, add_watermarker=False,
    ).to('cuda', silence_dtype_warnings=True)
    pipeline.enable_xformers_memory_efficient_attention()

    return pipeline


PIPELINE = load_pipeline()


@torch.inference_mode()
def handler(event):
    event_input = event['input']

    validated_input = validate(event_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    event_input = validated_input['validated_input']

    scheduler_name = event_input['scheduler']
    s3_presigned_post = event_input['s3_presigned_post']
    image_url = event_input['image']
    mask_url = event_input['mask_image']
    prompt = event_input['prompt']
    negative_prompt = event_input['negative_prompt']
    steps = event_input['steps']
    guidance_scale = event_input['guidance_scale']
    merge_padding = event_input['merge_padding']
    seed = event_input['seed']
    width = event_input['width']
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")

    image = download_image(image_url).convert('RGB')
    mask = download_image(mask_url).convert('RGB')

    ratio = image.size[0] / mask.size[1]
    height = round(width / ratio)
    height -= height % 8

    generator = torch.Generator("cuda").manual_seed(seed)

    PIPELINE.scheduler = make_scheduler(scheduler_name, PIPELINE.scheduler.config)

    kwargs = {
        'width': width,
        'height': height,
        'prompt': prompt,
        'image': image,
        'mask_image': mask,
        'negative_prompt': negative_prompt,
        'num_inference_steps': steps,
        'guidance_scale': guidance_scale,
        'num_images_per_prompt': 1,
        'generator': generator,
    }

    repainted_image = PIPELINE(**kwargs).images[0]
    repainted_image = repainted_image.resize(image.size)
    repainted_image = merge_images(image, mask, repainted_image, merge_padding)

    return upload_image(repainted_image, s3_presigned_post)


def merge_images(
        original_image: Image,
        mask_image: Image,
        repainted_image: Image,
        padding: float,
) -> Image:
    if padding == 1:
        return repainted_image

    mask_image_arr = np.array(mask_image.convert("L"))
    mask_image_arr = mask_image_arr[:, :, None]
    mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
    mask_image_arr[mask_image_arr < 0.5] = 0
    mask_image_arr[mask_image_arr >= 0.5] = 1

    if repainted_image.size != original_image.size:
        repainted_image = repainted_image.resize(original_image.size)

    mask_image_arr = cv2.GaussianBlur(mask_image_arr, (0, 0), 3, 3)
    mask_image_arr = cv2.merge([mask_image_arr, mask_image_arr, mask_image_arr])

    merged_image_arr = (1 - mask_image_arr) * original_image + mask_image_arr * repainted_image
    merged_image = Image.fromarray(merged_image_arr.round().astype("uint8"))

    return merged_image


def download_image(url: str) -> Image:
    resp = requests.get(url)
    if not resp.status_code == 200:
        raise RuntimeError(f'Could not download the image: {url}.')
    return Image.open(BytesIO(resp.content))


def upload_image(image: Image, s3_presigned_post) -> Optional[dict]:
    io_ = BytesIO()
    image.save(io_, 'png')
    io_.seek(0)
    resp = requests.post(
        s3_presigned_post['url'],
        files={
            'file': io_,
        },
        data={**s3_presigned_post['fields']},
    )

    if resp.status_code == 200:
        return {'status_code': 200}
    else:
        return {
            'status_code': resp.status_code,
            'error': resp.content
        }


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


runpod.serverless.start({"handler": handler})
