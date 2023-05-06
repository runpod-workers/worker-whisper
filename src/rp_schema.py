INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': True
    },
    'model': {
        'type': str,
        'required': False
    },
    'transcription': {
        'type': str,
        'required': False
    },
    'translate': {
        'type': bool,
        'required': False
    },
    'language': {
        'type': str,
        'required': False
    },
    'temperature': {
        'type': float,
        'required': False
    },
    'best_of': {
        'type': int,
        'required': False
    },
    'beam_size': {
        'type': int,
        'required': False
    },
    'patience': {
        'type': float,
        'required': False
    },
    'length_penalty': {
        'type': float,
        'required': False
    },
    'suppress_tokens': {
        'type': str,
        'required': False
    },
    'initial_prompt': {
        'type': str,
        'required': False
    },
    'condition_on_previous_text': {
        'type': bool,
        'required': False
    },
    'temperature_increment_on_fallback': {
        'type': float,
        'required': False
    },
    'compression_ratio_threshold': {
        'type': float,
        'required': False
    },
    'logprob_threshold': {
        'type': float,
        'required': False
    },
    'no_speech_threshold': {
        'type': float,
        'required': False
    }
}
