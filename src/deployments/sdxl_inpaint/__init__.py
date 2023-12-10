import torch
from diffusers import (
    AutoPipelineForInpainting,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

from deployments.sdxl import Deployment as SDXLDeployment
from deployments.sdxl_inpaint.schema import INPUT_SCHEMA

from utils import download_image


class Deployment(SDXLDeployment):
    MODEL_NAME = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
    MODEL_VARIANT = 'fp16'
    SCHEMA = INPUT_SCHEMA

    def download_model(self):
        return AutoPipelineForInpainting.from_pretrained(
            self.MODEL_NAME,
            variant=self.MODEL_VARIANT,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )

    def get_pipeline_kwargs(self, event_input):
        kwargs = super().get_pipeline_kwargs(event_input)

        mask = download_image(event_input['mask_image']).convert('RGB')

        width = event_input['width']
        ratio = mask.size[0] / mask.size[1]
        height = round(width / ratio)
        height -= height % 8

        kwargs.update({
            'mask': mask,
            'width': width,
            'height': height,
        })

        return kwargs
