import torch

from constants import MODEL_NAME, MODEL_VARIANT

from diffusers import AutoPipelineForInpainting


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    pipe = fetch_pretrained_model(
        AutoPipelineForInpainting,
        MODEL_NAME,
        variant=MODEL_VARIANT,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    return pipe


if __name__ == "__main__":
    fetch_pretrained_model(
        AutoPipelineForInpainting,
        MODEL_NAME,
        variant=MODEL_VARIANT,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
