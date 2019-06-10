import os, cv2, json
import torch
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
from PIL import Image
import random
import copy
# import torchaudio
import librosa
from scipy.io import wavfile
import math


class AudioSetDatasetTrain(Dataset):
    """
    AudioSet for Train
    """

    def __init__(self, config):
        super(AudioSetDatasetTrain, self).__init__()
        self.config = copy.deepcopy(config)
        self.data_dir = self.config.data.train_data_dir
        self.video_path = os.path.join(self.data_dir, 'video')
        self.audio_path = os.path.join(self.data_dir, 'audio')
        self.vid2genreFile = os.path.join(self.data_dir, 'vid2genre.json')
        with open(self.vid2genreFile, 'r') as fin:
            self.vid2genre = json.load(fin)
        self.audio_files = self.video_files = list(self.vid2genre.keys())
        self.imgSize = self.config.data.imgSize
        self.fps = self.config.data.fps
        self.time = self.config.data.v_time
        # the origin code sample all the frames in one video, i.e. self.frame_sample_s = 1
        self.frame_sample_s = self.config.data.frame_sample_s
        # Frames per video
        self.fpv = 1 + self.fps * (self.time - 1) // self.frame_sample_s
        # total frames of all the videos
        self.tot_frames = len(self.vid2genre) * self.fpv
        self.length = 2 * self.tot_frames

        # STFT params
        self.audRate = self.config.data.audRate
        self.audLen = self.config.data.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.stft_frame = self.config.data.stft_frame
        self.stft_hop = self.config.data.stft_hop
        self.HS = self.config.data.AudH
        self.WS = self.config.data.AudW

        self._vid_transform = self.get_video_transform()
        self._aud_transform = self.get_audio_transform

    def get_video_transform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize(int(self.config.data.imgSize * 1.1), Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(self.config.data.imgSize))
        transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))
        vid_transform = transforms.Compose(transform_list)
        return vid_transform

    def get_audio_transform(self, audio):
        """
        transform the audio, just scale the volume of the audio
        :param audio: audio in
        :return: audio out
        """
        scale = random.random() + 0.5  # 0.5-1.5
        audio = scale * audio
        return audio

    def __len__(self):
        # Consider all positive and negative examples
        return self.length

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            # audio_raw, rate = torchaudio.load(path)
            # audio_raw = audio_raw.numpy().astype(np.float32)
            #
            # # range to [-1, 1]
            # audio_raw *= (2.0 ** -31)
            #
            # # convert to mono
            # if audio_raw.shape[1] == 2:
            #     audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            # else:
            #     audio_raw = audio_raw[:, 0]
            audio_raw, rate = librosa.load(path, sr=None, mono=True)
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)
            # rate1, audio_raw1 = wavfile.read(path)

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)
        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate // self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen // 2 - (center - start): self.audLen // 2 + (end - center)] = \
            audio_raw[start:end]

        # randomize volume
        audio = self._aud_transform(audio)
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def __getitem__(self, idx):
        # Positive examples
        if idx < self.length / 2:
            result = [0]
            video_idx = int(idx / self.fpv)
            video_frame_number = idx % self.fpv
            frame_time = 500 + (video_frame_number * 1000 * self.frame_sample_s / self.fps)

            audio_file_path = os.path.join(self.audio_path, self.audio_files[video_idx] + '.wav')
            # Extract relevant audio file
            time = frame_time / 1000.0
            # Get video ID
            videoID = self.video_files[video_idx]
            vidClasses = self.vid2genre[videoID]

        # Negative examples
        else:
            result = [1]
            video_idx = int((idx - self.length / 2) / self.fpv)
            video_frame_number = (idx - self.length / 2) % self.fpv
            frame_time = 500 + (video_frame_number * 1000 * self.frame_sample_s / self.fps)

            # the original code sample the nagative audio according to the rest of the selected
            # video class, which i think is wrong, this is a self-supervised learning, we should not
            # sample the data by the class label, just randomly pick one
            random_audio_idx = (video_idx + random.randint(1, len(self.audio_files) - 1)) % len(self.audio_files)
            randomAudioID = self.audio_files[random_audio_idx]

            # Read the audio now
            audio_file_path = os.path.join(self.audio_path, randomAudioID + '.wav')
            time = (500 + (np.random.randint(self.fpv) * self.frame_sample_s * 1000 / self.fps)) / 1000.0

            # Get video ID
            videoID = self.video_files[video_idx]
            vidClasses = self.vid2genre[videoID]

        # Extract relevant frame
        #########################
        vidcap = cv2.VideoCapture(os.path.join(self.video_path, videoID + '.mp4'))
        vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
        success, image = vidcap.read()
        # Some problem with image, return some random stuff
        if image is None:
            ret_dict = {'image': torch.Tensor(np.random.rand(3, self.imgSize, self.imgSize)), 'audio': torch.Tensor(
                np.random.rand(1, self.HS, self.WS)), 'target': torch.LongTensor([2])}
            return ret_dict
        ##############################
        image = self._vid_transform(image)
        # select 1 sec
        # audio = self._load_audio(audio_file_path, time)
        # spectrogram, _ = self._stft(audio)
        rate, samples = wav.read(audio_file_path)
        start = int(time * self.audRate) - self.audLen // 2
        end = int(time * self.audRate) + self.audLen // 2
        samples = samples[start:end]
        frequencies, times, spectrogram = signal.spectrogram(samples, self.audRate, nperseg=512, noverlap=274)
        # Remove bad examples
        if spectrogram.shape != (self.HS, self.WS):
            ret_dict = {'image': torch.Tensor(np.random.rand(3, self.imgSize, self.imgSize)), 'audio': torch.Tensor(
                np.random.rand(1, self.HS, self.WS)), 'target': torch.LongTensor([2])}
            return ret_dict

        # Audio
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        audio = torch.Tensor(spectrogram.reshape(spec_shape))
        ret_dict = {'image': image, 'audio': audio, 'target': torch.LongTensor(result)}
        return ret_dict


class AudioSetDatasetVal(Dataset):
    """
    AudioSet for Train
    """

    def __init__(self, config):
        super(AudioSetDatasetVal, self).__init__()
        self.config = copy.deepcopy(config)
        self.data_dir = self.config.data.val_data_dir
        self.video_path = os.path.join(self.data_dir, 'video')
        self.audio_path = os.path.join(self.data_dir, 'audio')
        self.vid2genreFile = os.path.join(self.data_dir, 'vid2genre.json')
        with open(self.vid2genreFile, 'r') as fin:
            self.vid2genre = json.load(fin)
        self.audio_files = self.video_files = list(self.vid2genre.keys())
        self.imgSize = self.config.data.imgSize
        self.fps = self.config.data.fps
        self.time = self.config.data.v_time
        # the origin code sample all the frames in one video, i.e. self.frame_sample_s = 1
        self.frame_sample_s = self.config.data.frame_sample_s
        # Frames per video
        self.fpv = 1
        # total frames of all the videos
        self.tot_frames = len(self.vid2genre) * self.fpv
        self.length = self.tot_frames

        # STFT params
        self.audRate = self.config.data.audRate
        self.audLen = self.config.data.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.stft_frame = self.config.data.stft_frame
        self.stft_hop = self.config.data.stft_hop
        self.HS = self.config.data.AudH
        self.WS = self.config.data.AudW

        self._vid_transform = self.get_video_transform()
        self._aud_transform = self.get_audio_transform

    def get_video_transform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize(self.config.data.imgSize, Image.BICUBIC))
        transform_list.append(transforms.CenterCrop(self.config.data.imgSize))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))
        vid_transform = transforms.Compose(transform_list)
        return vid_transform

    def get_audio_transform(self, audio):
        return audio

    def __len__(self):
        # Consider all positive and negative examples
        return self.length

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            # audio_raw, rate = torchaudio.load(path)
            # audio_raw = audio_raw.numpy().astype(np.float32)
            #
            # # range to [-1, 1]
            # audio_raw *= (2.0 ** -31)
            #
            # # convert to mono
            # if audio_raw.shape[1] == 2:
            #     audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            # else:
            #     audio_raw = audio_raw[:, 0]
            audio_raw, rate = librosa.load(path, sr=None, mono=True)
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)
        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate // self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen // 2 - (center - start): self.audLen // 2 + (end - center)] = \
            audio_raw[start:end]

        # randomize volume
        audio = self._aud_transform(audio)
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def __getitem__(self, idx):
        # Positive examples
        result = [0]
        video_idx = int(idx / self.fpv)
        video_frame_number = idx % self.fpv
        frame_time = (self.time / 2) * 1000.0
        audio_file_path = os.path.join(self.audio_path, self.audio_files[video_idx] + '.wav')
        # Extract relevant audio file
        time = frame_time / 1000.0
        # Get video ID
        videoID = self.video_files[video_idx]
        vidClasses = self.vid2genre[videoID]

        # Extract relevant frame
        #########################
        vidcap = cv2.VideoCapture(os.path.join(self.video_path, videoID + '.mp4'))
        vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
        success, image = vidcap.read()
        # Some problem with image, return some random stuff
        if image is None:
            ret_dict = {'image': torch.Tensor(np.random.rand(3, self.imgSize, self.imgSize)), 'audio': torch.Tensor(
                np.random.rand(1, self.HS, self.WS)), 'target': torch.LongTensor([2])}
            return ret_dict
        ##############################
        image = self._vid_transform(image)
        # select 1 sec
        # audio = self._load_audio(audio_file_path, time)
        # spectrogram, _ = self._stft(audio)
        rate, samples = wav.read(audio_file_path)
        start = int(time * self.audRate) - self.audLen // 2
        end = int(time * self.audRate) + self.audLen // 2
        samples = samples[start:end]
        frequencies, times, spectrogram = signal.spectrogram(samples, self.audRate, nperseg=512, noverlap=274)

        # Remove bad examples
        if spectrogram.shape != (self.HS, self.WS):
            ret_dict = {'image': torch.Tensor(np.random.rand(3, self.imgSize, self.imgSize)), 'audio': torch.Tensor(
                np.random.rand(1, self.HS, self.WS)), 'target': torch.LongTensor([2])}
            return ret_dict

        # Audio
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        audio = torch.Tensor(spectrogram.reshape(spec_shape))
        ret_dict = {'image': image, 'audio': audio, 'target': torch.LongTensor(result)}
        return ret_dict


class AudioSetDatasetTest(Dataset):
    """
    AudioSet for Test
    """

    def __init__(self, config):
        super(AudioSetDatasetTest, self).__init__()
        self.config = copy.deepcopy(config)
        self.data_dir = self.config.data.test_data_dir
        self.video_path = os.path.join(self.data_dir, 'video')
        self.audio_path = os.path.join(self.data_dir, 'audio')
        self.vid2genreFile = os.path.join(self.data_dir, 'vid2genre.json')
        with open(self.vid2genreFile, 'r') as fin:
            self.vid2genre = json.load(fin)
        self.audio_files = self.video_files = list(self.vid2genre.keys())
        self.imgSize = self.config.data.imgSize
        self.fps = self.config.data.fps
        self.time = self.config.data.v_time
        # the origin code sample all the frames in one video, i.e. self.frame_sample_s = 1
        self.frame_sample_s = self.config.data.frame_sample_s
        # Frames per video
        self.fpv = 1
        # total frames of all the videos
        self.tot_frames = len(self.vid2genre) * self.fpv
        self.length = self.tot_frames

        # STFT params
        self.audRate = self.config.data.audRate
        self.audLen = self.config.data.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.stft_frame = self.config.data.stft_frame
        self.stft_hop = self.config.data.stft_hop
        self.HS = self.config.data.AudH
        self.WS = self.config.data.AudW

        self._vid_transform = self.get_video_transform()
        self._aud_transform = self.get_audio_transform

    def get_video_transform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize(self.config.data.imgSize, Image.BICUBIC))
        transform_list.append(transforms.CenterCrop(self.config.data.imgSize))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))
        vid_transform = transforms.Compose(transform_list)
        return vid_transform

    def get_audio_transform(self, audio):
        """
        transform the audio, just scale the volume of the audio
        :param audio: audio in
        :return: audio out
        """
        return audio

    def __len__(self):
        # Consider all positive and negative examples
        return self.length

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            # audio_raw, rate = torchaudio.load(path)
            # audio_raw = audio_raw.numpy().astype(np.float32)
            #
            # # range to [-1, 1]
            # audio_raw *= (2.0 ** -31)
            #
            # # convert to mono
            # if audio_raw.shape[1] == 2:
            #     audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            # else:
            #     audio_raw = audio_raw[:, 0]
            audio_raw, rate = librosa.load(path, sr=None, mono=True)
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)
        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate // self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen // 2 - (center - start): self.audLen // 2 + (end - center)] = \
            audio_raw[start:end]

        # randomize volume
        audio = self._aud_transform(audio)
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def __getitem__(self, idx):
        result = [0]
        video_idx = int(idx / self.fpv)
        video_frame_number = idx % self.fpv
        frame_time = (self.time / 2) * 1000.0
        audio_file_path = os.path.join(self.audio_path, self.audio_files[video_idx] + '.wav')
        # Extract relevant audio file
        time = frame_time / 1000.0
        # Get video ID
        videoID = self.video_files[video_idx]
        vidClasses = self.vid2genre[videoID]

        # Extract relevant frame
        #########################
        vidcap = cv2.VideoCapture(os.path.join(self.video_path, videoID + '.mp4'))
        vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
        success, image = vidcap.read()
        # Some problem with image, return some random stuff
        if image is None:
            ret_dict = {'image': torch.Tensor(np.random.rand(3, self.imgSize, self.imgSize)), 'audio': torch.Tensor(
                np.random.rand(1, self.HS, self.WS)), 'target': torch.LongTensor([2]), 'vidClasses': vidClasses}
            return ret_dict
        ##############################
        image = self._vid_transform(image)
        # select 1 sec
        # audio = self._load_audio(audio_file_path, time)
        # spectrogram, _ = self._stft(audio)
        rate, samples = wav.read(audio_file_path)
        start = int(time * self.audRate) - self.audLen // 2
        end = int(time * self.audRate) + self.audLen // 2
        samples = samples[start:end]
        frequencies, times, spectrogram = signal.spectrogram(samples, self.audRate, nperseg=512, noverlap=274)

        # Remove bad examples
        if spectrogram.shape != (self.HS, self.WS):
            ret_dict = {'image': torch.Tensor(np.random.rand(3, self.imgSize, self.imgSize)), 'audio': torch.Tensor(
                np.random.rand(1, self.HS, self.WS)), 'target': torch.LongTensor([2])}
            return ret_dict

        # Audio
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        audio = torch.Tensor(spectrogram.reshape(spec_shape))
        ret_dict = {'image': image, 'audio': audio, 'target': torch.LongTensor(result), 'vidClasses': vidClasses}
        return ret_dict
