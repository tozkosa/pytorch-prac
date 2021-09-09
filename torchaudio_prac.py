import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import os

ECHO_DATA_PATH = "/home/tomoko/daon/nagoya2021/nagoya_20210727_cutout_5ms/"
PLACE = "crack_1"
TRAIN_TEST = "test"
HAMMER = "small"


if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    wav_path = os.path.join(ECHO_DATA_PATH, PLACE, TRAIN_TEST, HAMMER)
    print(wav_path)

    wav_files = []
    for f in os.listdir(wav_path):
        wav_files.append(f)

    print(len(wav_files))
    metadata = torchaudio.info(os.path.join(wav_path, wav_files[0]))
    print(metadata)
    

    