from runpod.serverless.utils.rp_validator import validate

import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from deployments.sdxl.schema import INPUT_SCHEMA

from utils import download_image, upload_image, merge_images


class Deployment:
    MODEL_NAME = 'stabilityai/stable-diffusion-xl-refiner-1.0'
    MODEL_VARIANT = 'fp16'
    SCHEMA = INPUT_SCHEMA
    SCHEDULERS = {
        "PNDM": PNDMScheduler,
        "KLMS": LMSDiscreteScheduler,
        "DDIM": DDIMScheduler,
        "K_EULER": EulerDiscreteScheduler,
        "DPMSolverMultistep": DPMSolverMultistepScheduler,
    }

    def __init__(self):
        self.pipeline = None

    def load_pipeline(self):
        pipeline = self.download_model().to(
            'cuda',
            silence_dtype_warnings=True,
        )
        pipeline.enable_xformers_memory_efficient_attention()

        self.pipeline = pipeline

    def download_model(self):
        return StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.MODEL_NAME,
            variant=self.MODEL_VARIANT,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )

    def get_scheduler(self, name, config):
        return self.SCHEDULERS[name].from_config(config)

    def get_pipeline_kwargs(self, event_input):
        image = download_image(event_input['image']).convert('RGB')

        kwargs = {
            'prompt': event_input['prompt'],
            'image': image,
            'negative_prompt': event_input['negative_prompt'],
            'num_inference_steps': event_input['steps'],
            'guidance_scale': event_input['guidance_scale'],
            'num_images_per_prompt': 1,
        }

        generator = self.get_generator(seed=event_input['seed'])
        if generator:
            kwargs['generator'] = generator

        return kwargs

    def get_generator(self, seed):
        if seed is not None:
            return torch.Generator("cuda").manual_seed(seed)

    def run_pipeline(self, **kwargs):
        return self.pipeline(**kwargs).images[0]

    def handle_event(self, event):
        event_input = event['input']

        validated_input = validate(event_input, self.SCHEMA)

        if 'errors' in validated_input:
            return {"error": validated_input['errors']}
        event_input = validated_input['validated_input']

        self.pipeline.scheduler = self.get_scheduler(
            name=event_input['scheduler'],
            config=self.pipeline.scheduler.config,
        )

        kwargs = self.get_pipeline_kwargs(event_input)

        repainted_image = self.run_pipeline(**kwargs)

        repainted_image = repainted_image.resize(kwargs['image'].size)
        repainted_image = merge_images(
            kwargs['image'],
            kwargs['mask_image'],
            repainted_image,
            event_input['merge_padding'],
        )

        return upload_image(repainted_image, event_input['s3_presigned_post'])
