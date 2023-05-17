"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np

from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES
from whisper.utils import format_timestamp


class Predictor:
    ''' A Predictor class for the Whisper model '''

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.models = {}

        def load_model(model_name):
            '''
            Load the model from the weights folder.
            '''
            try:
                with open(f"weights/{model_name}.pt", "rb") as model_file:
                    checkpoint = torch.load(model_file, map_location="cpu")
                    dims = ModelDimensions(**checkpoint["dims"])
                    model = Whisper(dims)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    return model_name, model
            except FileNotFoundError:
                print(f"Model {model_name} could not be found.")
                return None, None

        model_names = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain text",
        translate=False,
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=None,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    ):
        """Run a single prediction on the model"""
        print(f"Transcribe with {model_name} model")
        model = self.models[model_name]
        if torch.cuda.is_available():
            model = model.to("cuda")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        args = {
            "language": language,
            "best_of": best_of,
            "beam_size": beam_size,
            "patience": patience,
            "length_penalty": length_penalty,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
        }

        result = model.transcribe(str(audio), temperature=temperature, **args)

        if transcription == "plain text":
            transcription = result["text"]
        elif transcription == "srt":
            transcription = write_srt(result["segments"])
        else:
            transcription = write_vtt(result["segments"])

        if translate:
            translation = model.transcribe(
                str(audio), task="translate", temperature=temperature, **args
            )

        return {
            "segments": result["segments"],
            "detected_language": LANGUAGES[result["language"]],
            "transcription": transcription,
            "translation": translation["text"] if translate else None,
        }


def write_vtt(transcript):
    '''
    Write the transcript in VTT format.
    '''
    result = ""
    for segment in transcript:
        result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result


def write_srt(transcript):
    '''
    Write the transcript in SRT format.
    '''
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result
