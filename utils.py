import numpy as np
import pandas
from pandas import DataFrame, Series
from scipy.signal import savgol_filter
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from pydub import AudioSegment
import faiss
from sentence_transformers import SentenceTransformer
import dtw

import math


NAN = -1

sentences = pandas.read_csv('data/sentences.csv')

print("model loading...")
model = SentenceTransformer("jhgan/ko-sroberta-multitask")
encoded_data = model.encode(sentences["sentence"])
print("model loaded")


def convert_audio_for_model(user_file, output_file='converted_audio_file.wav', sampling_rate=16000):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(sampling_rate).set_channels(1)
  audio.export(output_file, format='wav')
  return output_file


def smooth_data_savgol_2(arr, span):  
    return savgol_filter(arr, span, 2)


def smooth_data_fft(arr, span):
    w = fftpack.rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return fftpack.irfft(w)


def smooth(graph):
    # pitch_y 데이터를 smoothing한다.
    smoothed_pitch_y = smooth_data_fft(graph["pitch_y"], 1.2)
    smoothed_pitch_y = smoothed_pitch_y.tolist()
    graph["pitch_y"] = smoothed_pitch_y
    # print(graph)
    return graph


def fill_gap(graph, fill_with = NAN):
    # pitch_x가 빈 값 없이 연속적인 정수 값을 갖도록 바꾸고,
    # pitch_y 데이터 중 비어있는 값을 fill_with로 채운다.
    pitch_x, pitch_y = graph["pitch_x"], graph["pitch_y"]

    if len(pitch_x) == 0 and len(pitch_y) == 0:
        return None

    pitch_x_start_value = pitch_x[0]
    pitch_x = [x - pitch_x_start_value for x in pitch_x]

    pitch_x_last_value = pitch_x[-1]
    filled_pitch_x = [i for i in range(pitch_x_last_value + 1)]
    filled_pitch_y = [fill_with] * (pitch_x_last_value + 1)

    for i, x in enumerate(pitch_x):
        filled_pitch_y[x] = pitch_y[i]

    graph["pitch_x"] = filled_pitch_x
    graph["pitch_y"] = filled_pitch_y

    return graph


def scale(graph, target_length):
    # 더 짧은 그래프의 길이를 더 긴 그래프의 길이에 맞춰서 pitch_x의 값들을 scale한다.

    if graph is None or len(graph["pitch_x"]) == 0 or len(graph["pitch_y"]) == 0:
        return None
    if target_length < len(graph["pitch_x"]):
        return graph
    
    scale_factor = (target_length - 1) / (len(graph["pitch_x"]) - 1)
    graph["pitch_x"] = [math.ceil(x * scale_factor) for x in graph["pitch_x"]]
    graph = fill_gap(graph)

    return graph


def interpolate(graph, target=[ NAN ], method="values"):
    # target으로 채워진 값을 보간한다.

    ts = Series(graph["pitch_y"], index=graph["pitch_x"])

    ts.replace(target, np.nan, inplace=True)
    ts.interpolate(method=method, inplace=True)

    graph["pitch_y"] = ts.tolist()
    
    return graph
    

def preprocess(graph_1, graph_2):
    graph_1 = fill_gap(graph_1)
    graph_2 = fill_gap(graph_2)

    shorter_graph, longer_graph = sorted([graph_1, graph_2], key=lambda x: len(x["pitch_x"]))

    target_length = len(longer_graph["pitch_x"])
    shorter_graph = scale(shorter_graph, target_length)

    shorter_graph = interpolate(shorter_graph)
    longer_graph = interpolate(longer_graph)

    return shorter_graph, longer_graph


def get_MAPE_score(graph_1, graph_2):
    if len(graph_1["pitch_x"]) != len(graph_2["pitch_x"]):
        raise ValueError("The length of two graphs must be same.")
    
    target_y = DataFrame(graph_1["pitch_y"] if graph_1["label"] == "target" else graph_2["pitch_y"])
    user_y = DataFrame(graph_1["pitch_y"] if graph_1["label"] == "user" else graph_2["pitch_y"])

    MAPE = np.mean(np.abs((target_y - user_y) / target_y)) * 100
    score = 100 - MAPE[0]
    # print("MAPE: ", MAPE)
    return score


def get_DTW_score(graph_1, graph_2):
    score = dtw.dtw(graph_1["pitch_y"], graph_2["pitch_y"], keep_internals=True)
    return score.distance


def compare(target, user):
    # DTW_score = get_DTW_score(target, user)
    graph_1, graph_2 = preprocess(target, user)
    MAPE_score = get_MAPE_score(graph_1, graph_2)
    DTW_score = get_DTW_score(graph_1, graph_2)
    # print("DTW: ", DTW_score)
    # print("MAPE: ", MAPE_score)
    return MAPE_score, DTW_score


def draw_graph(graph):
    # resize graph
    plt.figure(figsize=(14, 8))
    # graph['pitch_y'] = [ np.nan if x == -1.0 else x for x in graph['pitch_y']]
    plt.plot(graph["pitch_x"], graph["pitch_y"], label="y")
    smoothed_y = smooth(graph)["pitch_y"]
    plt.plot(graph["pitch_x"], smoothed_y, label="smoothed_y")

    plt.legend()
    plt.show()


def create_id_to_sen_dict():
    id_to_sen = {}
    for i in range(len(sentences)):
        id_to_sen[sentences["id"][i]] = sentences["sentence"][i]
    return id_to_sen


def create_sen_to_id_dict():
    sen_to_id = {}
    for i in range(len(sentences)):
        sen_to_id[sentences["sentence"][i]] = sentences["id"][i]
    return sen_to_id


def semantic_sentence_search(text: str, n: int = 3):
    id_to_sen = create_id_to_sen_dict()

    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(encoded_data, np.array(sentences["id"]))

    faiss.write_index(index, 'sentences.index')

    query_vector = model.encode([text])
    top_n = index.search(query_vector, n)

    return {
        str(i): {
            "id": int(top_n[1][0][i]),
            "sentence": id_to_sen[int(top_n[1][0][i])],
        } for i in range(n)
    }
