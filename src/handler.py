import runpod
import torch

from utils import get_deployment


torch.cuda.empty_cache()


deployment = get_deployment()
deployment.load_pipeline()


@torch.inference_mode()
def handler(event):
    return deployment.handle_event(event)


runpod.serverless.start({"handler": handler})
