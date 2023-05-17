''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from rp_schema import INPUT_VALIDATIONS

MODEL = predict.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Setting the float parameters
    job_input['temperature'] = float(job_input.get('temperature', 0))
    job_input['patience'] = float(job_input.get('patience', 0))
    job_input['length_penalty'] = float(job_input.get('length_penalty', 0))
    job_input['temperature_increment_on_fallback'] = float(
        job_input.get('temperature_increment_on_fallback', 0.2)
    )
    job_input['compression_ratio_threshold'] = float(
        job_input.get('compression_ratio_threshold', 2.4)
    )
    job_input['logprob_threshold'] = float(job_input.get('logprob_threshold', -1.0))
    job_input['no_speech_threshold'] = 0.6

    # Input validation
    validated_input = validate(job_input, INPUT_VALIDATIONS)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}

    job_input['audio'] = download_files_from_urls(job['id'], [job_input['audio']])[0]

    whisper_results = MODEL.predict(
        audio=job_input["audio"],
        model_name=job_input.get("model", 'base'),
        transcription=job_input.get('transcription', 'plain_text'),
        translate=job_input.get('translate', False),
        language=job_input.get('language', None),
        temperature=job_input["temperature"],
        best_of=job_input.get("best_of", 5),
        beam_size=job_input.get("beam_size", 5),
        patience=job_input["patience"],
        length_penalty=job_input["length_penalty"],
        suppress_tokens=job_input.get("suppress_tokens", "-1"),
        initial_prompt=job_input.get('initial_prompt', None),
        condition_on_previous_text=job_input.get('condition_on_previous_text', True),
        temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
        compression_ratio_threshold=job_input["compression_ratio_threshold"],
        logprob_threshold=job_input["logprob_threshold"],
        no_speech_threshold=job_input["no_speech_threshold"],
    )

    rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run})
