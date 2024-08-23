import cv2
import numpy as np
import librosa

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    try:
        y, sr = librosa.load(audio_path, sr=None)
        n_fft = 2048
        hop_length = 512
        fmax = 22000
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=fmax)
        spec_db = librosa.power_to_db(spec)
        height,width= spec_db.shape
        width_avg = abs(np.sum(np.sum(spec_db,axis = 1))/width)
        class_name = 'metal' if width_avg < 3500 else 'cardboard'
        return class_name
    except:
        return 'cardboard'
