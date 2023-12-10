DEFAULT_NEGATIVE_PROMPT = (
    'out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, '
    'ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, '
    'poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, '
    'bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, '
    'missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, '
    'long neck, username, watermark, signature'
)


INPUT_SCHEMA = {
    's3_presigned_post': {
        'type': dict,
        'required': True,
    },
    'image': {
        'type': str,
        'required': True,
    },
    'prompt': {
        'type': str,
        'required': True,
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024,
        'constraints': lambda width: width % 8 == 0,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': DEFAULT_NEGATIVE_PROMPT,
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'DDIM'
    },
    'steps': {
        'type': int,
        'required': False,
        'default': 25
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
}