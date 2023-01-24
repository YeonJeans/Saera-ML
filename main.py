from fastapi import FastAPI, status, HTTPException, Response

from dotenv import load_dotenv

import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

from pydub import AudioSegment

import logging
import math
import statistics
import httpx
import os
import urllib

logger = logging.getLogger()
logger.setLevel(logging.INFO)

print("TensorFlow version: ", tf.__version__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(verbose=True)
sampling_rate = os.getenv('SAMPLING_RATE')

app = FastAPI()


def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
  sampling_rate = os.getenv('SAMPLING_RATE')
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(sampling_rate).set_channels(1)
  audio.export(output_file, format='wav')
  return output_file


@app.get('/tts')
async def get_tts_wav_from_clova(text = None):
    if text is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='text is required')
    
    enc_text = urllib.parse.quote(text)
    clova_voice_api_url = os.getenv('CLOVA_VOICE_API_URL')
    key_id = os.getenv('X_NCP_APIGW_API_KEY_ID')
    secret_key = os.getenv('X_NCP_APIGW_API_KEY')
    speaker = os.getenv('SPEAKER')
    output_format = os.getenv('FORMAT')

    data = "speaker={}&volume=0&speed=0&pitch=0&format={}&text={}&sampling-rate={}".format(speaker, output_format, enc_text, sampling_rate)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(clova_voice_api_url, data=data.encode('utf-8'), headers={
            'X-NCP-APIGW-API-KEY-ID': key_id,
            'X-NCP-APIGW-API-KEY': secret_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        })
    
    res_code = response.status_code
    if (res_code != 200):
        logger.error('[/get] CLOVA Voice API Error code: {}'.format(res_code))
        raise HTTPException(status_code=res_code, detail='Error code: {}'.format(res_code))
    
    logger.info('[/get] TTS wav 저장')
    response_body = response.content
    with open('audio/{}.wav'.format(text), 'wb') as f:
        f.write(response_body)
    
    return Response(content=response_body, media_type='audio/x-www-form-urlencoded')


@app.get('/pitch-graph')
def get_pitch_graph(audio_file = None):
    if audio_file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='audio file(wav) is required')
    
    duration = len(audio_file) / sampling_rate
    print(f'Sample rate: {sample_rate} Hz')
    print(f'Total duration: {duration:.2f}s')
    print(f'Size of the input: {len(audio_samples)}')


@app.get('/score')
def calculate_score(text = None):
    pass
    