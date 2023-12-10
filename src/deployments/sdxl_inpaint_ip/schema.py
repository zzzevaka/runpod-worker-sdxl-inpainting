from deployments.sdxl_inpaint.schema import INPUT_SCHEMA as SDXL_INPAINT_INPUT_SCHEMA


INPUT_SCHEMA = {
    **SDXL_INPAINT_INPUT_SCHEMA,
    'image_prompt': {
        'type': str,
        'required': True,
    },
}
