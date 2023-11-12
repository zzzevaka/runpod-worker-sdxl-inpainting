import os
from typing import Optional
from io import BytesIO

import requests
import cv2
import numpy as np
from PIL import Image


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


def get_deployment():
    if os.getenv('DEPLOYMENT', 'sdxl') == 'sdxl_ip':
        from deployments.sdxl_ip import SDXLIPInpaintDeployment
        return SDXLIPInpaintDeployment()

    from deployments.sdxl import SDXLInpaintDeployment
    return SDXLInpaintDeployment()
