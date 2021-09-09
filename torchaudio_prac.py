import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import os
import librosa
from matplotlib import pyplot as plt

# ECHO_DATA_PATH = "/home/tomoko/daon/nagoya2021/nagoya_20210727_cutout_5ms/"
ECHO_DATA_PATH = "D:\daon_data\\nagoya_20210727_cutout_5ms"
PLACE = "crack_1"
TRAIN_TEST = "test"
HAMMER = "small"

def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)

    if sample_rate:
        print("Sample Rate:", sample_rate)
    print(f"Shape:", tuple(waveform.shape))
    print(f"Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    # print(waveform)
    print()

def plot_waveform(waveform, sample_rate, title="waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    # print(time_axis)

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)




if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    wav_path = os.path.join(ECHO_DATA_PATH, PLACE, TRAIN_TEST, HAMMER)
    print(wav_path)

    wav_files = []
    for f in os.listdir(wav_path):
        wav_files.append(f)

    print(len(wav_files))

    # The following does not work.
    # metadata = torchaudio.SignalInfo(os.path.join(wav_path, wav_files[0]))
    # print(metadata)

    # for i in range(0, 10, 2):
    #     file_path = os.path.join(wav_path, wav_files[i])
    #     waveform, sample_rate = torchaudio.load(file_path)
    #     print_stats(waveform, sample_rate=sample_rate)
    #     # plot_waveform(waveform, sample_rate)
    #     plot_specgram(waveform, sample_rate)
    # plt.show()

    file_path = os.path.join(wav_path, wav_files[0])
    waveform, sample_rate = torchaudio.load(file_path)

    # spectrogram

    # n_fft = 64
    # win_length = None
    # hop_length = 32

    # spectrogram = T.Spectrogram(
    #     n_fft=n_fft,
    #     win_length=win_length,
    #     hop_length=hop_length,
    #     power=2.0
    # )

    # spec = spectrogram(waveform)
    # print_stats(spec)
    # plot_spectrogram(spec[0], title='torchaudio')
    # print(wav_files[400])

    """MelSpectrogram"""
    # n_fft = 64
    # win_length = None
    # hop_length = 32
    # n_mels = 16

    # mel_spectrogram = T.MelSpectrogram(
    #     sample_rate=sample_rate,
    #     n_fft=n_fft,
    #     win_length=win_length,
    #     hop_length=hop_length,
    #     # center=True,
    #     # pad_mode="reflect",
    #     power=2.0,
    #     # norm='slaney',
    #     # onesided=True,
    #     n_mels=n_mels,
    #     # mel_scale="htk",
    # )

    # melspec = mel_spectrogram(waveform)
    # plot_spectrogram(
    #     melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq'
    # )

    # melspec_librosa = librosa.feature.melspectrogram(
    #     waveform.numpy()[0],
    #     sr=sample_rate,
    #     n_fft=n_fft,
    #     hop_length=hop_length,
    #     win_length=win_length,
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     n_mels=n_mels,
    #     norm='slaney',
    #     htk=True,
    # )
    # plot_spectrogram(
    #     melspec_librosa, title="MelSpectrogram - librosa", ylabel='mel freq')

    # mse = torch.square(melspec - melspec_librosa).mean().item()
    # print('Mean Square Difference: ', mse)

    """MFCC"""
    n_fft = 64
    win_length = None
    hop_length = 32
    n_mels = 16
    n_mfcc = 16

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            # 'n_ftt': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length,
            # 'mel_scale': 'htk'
        }
    )

    mfcc = mfcc_transform(waveform)

    plot_spectrogram(mfcc[0])


    plt.show()
    

    