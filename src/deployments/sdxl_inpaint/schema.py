from deployments.sdxl.schema import INPUT_SCHEMA as SDXL_INPUT_SCHEMA


INPUT_SCHEMA = {
    **SDXL_INPUT_SCHEMA,
    'mask_image': {
        'type': str,
        'required': True,
    },
    'merge_padding': {
        'type': float,
        'required': False,
        'default': 0.02,
        'constraints': lambda padding: 1 >= padding >= 0,
    }
}
