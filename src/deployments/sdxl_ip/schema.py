from deployments.sdxl.schema import INPUT_SCHEMA as SDXL_INPUT_SCHEMA


INPUT_SCHEMA = {
    'image_prompt': {
        'type': str,
        'required': True,
    },
    **SDXL_INPUT_SCHEMA,
}
