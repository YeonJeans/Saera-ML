from fastapi import FastAPI, status, HTTPException, Response, File, UploadFile

from dotenv import load_dotenv

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from pydub import AudioSegment

import logging
import httpx
import os
import urllib
from scipy.io import wavfile
import datetime


logger = logging.getLogger()
logger.setLevel(logging.INFO)

print("TensorFlow version: ", tf.__version__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(verbose=True)
sampling_rate = int(os.getenv('SAMPLING_RATE'))

os.environ["TFHUB_CACHE_DIR"] = ".cache/tfhub"
model = hub.load('https://tfhub.dev/google/spice/2')

app = FastAPI()


def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(sampling_rate).set_channels(1)
  audio.export(output_file, format='wav')
  return output_file


@app.get('/tts')
async def get_tts_wav_from_clova(text = None):
    logger.info('[/tts] called; text: {}'.format(text))

    if text is None:
        logger.error('[/tts] an error occurred; text is required')
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
        logger.error('[/tts] CLOVA Voice API Error code: {}'.format(res_code))
        raise HTTPException(status_code=res_code, detail='Error code: {}'.format(res_code))
    
    logger.info('[/tts] TTS wav 저장. text: {}'.format(text))
    response_body = response.content
    with open('audio/{}.wav'.format(text), 'wb') as f:
        f.write(response_body)

    return Response(content=response_body, media_type='audio/x-www-form-urlencoded')


@app.post('/pitch-graph')
def get_pitch_graph(audio: UploadFile = File(...)):
    logger.info('[/pitch-graph] called')

    if audio.file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='audio file(wav) is required')
    
    raw_audio_file = audio.file.read()
    random_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    with open('audio_pitch/{}.wav'.format(random_name), 'wb') as f:
        f.write(raw_audio_file)

    converted_audio_file = convert_audio_for_model('audio_pitch/{}.wav'.format(random_name), 'audio_pitch/converted_{}.wav'.format(random_name))
    _, audio_file = wavfile.read(converted_audio_file, 'rb')

    model_output = model.signatures["serving_default"](tf.constant(audio_file, tf.float32))

    pitch_outputs = model_output["pitch"]
    uncertainty_outputs = model_output["uncertainty"]

    # confidence = 1 - uncertainty
    confidence_outputs = list(1.0 - uncertainty_outputs)
    pitch_outputs = [ float(x) for x in pitch_outputs ]

    indices = range(len(pitch_outputs))

    # confidence 0.9 이상인 것만 추출
    confident_pitch_outputs = [ (i, p) for i, p, c in zip(indices, pitch_outputs, confidence_outputs) if c > 0.9 ]
    confident_pitch_outputs_x, confident_pitch_outputs_y = zip(*confident_pitch_outputs)

    response_body = {
        'pitch_x': confident_pitch_outputs_x,
        'pitch_y': confident_pitch_outputs_y
    }

    return response_body


@app.get('/score')
def calculate_pitch_score(text = None):
    pass