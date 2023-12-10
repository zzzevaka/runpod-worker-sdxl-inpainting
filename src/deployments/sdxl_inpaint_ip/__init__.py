import os.path
import logging
from urllib import request

from ip_adapter import IPAdapterXL

from deployments.sdxl_inpaint import Deployment as SDXLInpaintDeployment
from deployments.sdxl_inpaint_ip.schema import INPUT_SCHEMA

from utils import download_image


logger = logging.getLogger(__name__)


IP_ADAPTER_DIR = '/app/ip_adapter_models'
IP_ADAPTER_MODEL_NAME = 'ip-adapter_sdxl_vit-h.bin'
IP_ADAPTER_FILES = (
    (
        'https://huggingface.co/InvokeAI/ip_adapter_sdxl_vit_h/resolve/main/ip_adapter.bin',
        f'{IP_ADAPTER_DIR}/{IP_ADAPTER_MODEL_NAME}',
    ),
    (
        'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin',
        f'{IP_ADAPTER_DIR}/image_encoder/pytorch_model.bin'
    ),
    (
        'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json',
        f'{IP_ADAPTER_DIR}/image_encoder/config.json'
    ),
)


class Deployment(SDXLInpaintDeployment):
    SCHEMA = INPUT_SCHEMA

    def __init__(self):
        super().__init__()
        self.ip_adapter = None

    def load_pipeline(self):
        super().load_pipeline()
        self.ip_adapter = IPAdapterXL(
            self.pipeline,
            f'{IP_ADAPTER_DIR}/image_encoder',
            f'{IP_ADAPTER_DIR}/{IP_ADAPTER_MODEL_NAME}',
            'cuda',
            num_tokens=4,
        )

    def download_model(self):
        pipeline = super().download_model()

        if not os.path.isdir(IP_ADAPTER_DIR):
            os.mkdir(IP_ADAPTER_DIR)

        for file_url, file_path in IP_ADAPTER_FILES:
            if not os.path.isfile(file_path):
                logger.info('downloading: %s', file_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                request.urlretrieve(file_url, file_path)
                logger.info('finished')

        return pipeline

    def get_pipeline_kwargs(self, event_input):
        kwargs = super().get_pipeline_kwargs(event_input)

        image_prompt = download_image(event_input['image_prompt']).convert('RGB')
        image_prompt = image_prompt.resize((256, 256))

        kwargs['seed'] = event_input['seed']
        kwargs['num_samples'] = event_input['num_images']
        kwargs['pil_image'] = image_prompt

        return kwargs

    def get_generator(self, seed):
        return None

    def run_pipeline(self, **kwargs):
        return self.ip_adapter.generate(**kwargs)[0]
