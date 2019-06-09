import os, cv2, json
import torch
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import random
import copy

from . import video_transforms as vtransforms


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
        self.fps = self.config.data.fps
        self.time = self.config.data.v_time
        # the origin code sample all the frames in one video, i.e. self.frame_sample_s = 1
        self.frame_sample_s = self.config.data.frame_sample_s
        # Frames per video
        self.fpv = 1 + self.fps * (self.time - 1) // self.frame_sample_s
        # total frames of all the videos
        self.tot_frames = len(self.vid2genre) * self.fpv
        self.length = 2 * self.tot_frames

        self._vid_transform = self.get_video_transform()
        self._aud_transform = self.get_audio_transform

    def get_video_transform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_list.append(vtransforms.Resize(int(self.config.data.imgSize * 1.1), Image.BICUBIC))
        transform_list.append(vtransforms.RandomCrop(self.config.data.imgSize))
        transform_list.append(vtransforms.RandomHorizontalFlip())

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        vid_transform = transforms.Compose(transform_list)
        return vid_transform

    def get_audio_transform(self, audio):
        """
        transform the audio, just scale the volume of the audio
        :param audio: audio in
        :return: audio out
        """
        scale = random.random() + 0.5  # 0.5-1.5
        audio *= scale
        return audio

    def __len__(self):
        # Consider all positive and negative examples
        return self.length

    def __getitem__(self, idx):
        # Positive examples
        if idx < self.length / 2:
            result = [0]
            video_idx = int(idx / self.fpv)
            video_frame_number = idx % self.fpv
            frame_time = 500 + (video_frame_number * 1000 * self.frame_sample_s / self.fps)

            rate, samples = wav.read(os.path.join(self.audio_path, self.audio_files[video_idx] + '.wav'))
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
            rate, samples = wav.read(os.path.join(self.audio_path, randomAudioID + '.wav'))
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
            ret_dict = {'image': torch.Tensor(np.random.rand(3, 224, 224)), 'audio': torch.Tensor(
                np.random.rand(1, 257, 200)), 'target': torch.LongTensor([2])}
            return ret_dict
        ##############################
        # Bring the channel to front
        image = image.transpose(2, 0, 1)
        image = self._vid_transform(torch.Tensor(image))
        # select 1 sec
        start = int(time * 48000) - 24000
        end = int(time * 48000) + 24000
        samples = samples[start:end]
        samples = self._aud_transform(samples)
        frequencies, times, spectrogram = signal.spectrogram(samples, self.config.data.sampleRate, nperseg=512,
                                                             noverlap=274)

        # Remove bad examples
        if spectrogram.shape != (257, 200):
            return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(
                np.random.rand(1, 257, 200)), torch.LongTensor([2])

        # Audio
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        audio = torch.Tensor(spectrogram.reshape(spec_shape))
        ret_dict = {'image': image, 'audio': audio, 'target': torch.LongTensor(result)}
        return ret_dict


class AudioSetDatasetVal(Dataset):
    """
    AudioSet for Validation
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
        self.fps = self.config.data.fps
        self.time = self.config.data.v_time
        # the origin code sample all the frames in one video, i.e. self.frame_sample_s = 1
        self.frame_sample_s = self.config.data.frame_sample_s
        # Frames per video
        self.fpv = 1 + self.fps * (self.time - 1) // self.frame_sample_s
        # total frames of all the videos
        self.tot_frames = len(self.vid2genre) * self.fpv
        self.length = self.tot_frames * 2

        self._vid_transform = self.get_video_transform()
        self._aud_transform = self.get_audio_transform

    def get_video_transform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_list.append(vtransforms.Resize(self.config.data.imgSize, Image.BICUBIC))
        transform_list.append(vtransforms.CenterCrop(self.config.data.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
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

    def __getitem__(self, idx):
        # Positive examples
        if idx < self.length / 2:
            result = [0]
            video_idx = int(idx / self.fpv)
            video_frame_number = idx % self.fpv
            frame_time = 500 + (video_frame_number * 1000 * self.frame_sample_s / self.fps)

            rate, samples = wav.read(os.path.join(self.audio_path, self.audio_files[video_idx] + '.wav'))
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
            rate, samples = wav.read(os.path.join(self.audio_path, randomAudioID + '.wav'))
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
            ret_dict = {'image': torch.Tensor(np.random.rand(3, 224, 224)),
                        'audio': torch.Tensor(np.random.rand(1, 257, 200)),
                        'target': torch.LongTensor([2]), 'vidClasses': vidClasses}
            return ret_dict
        ##############################
        # Bring the channel to front
        image = image.transpose(2, 0, 1)
        image = self._vid_transform(torch.Tensor(image))
        # select 1 sec
        start = int(time * 48000) - 24000
        end = int(time * 48000) + 24000
        samples = samples[start:end]
        samples = self._aud_transform(samples)
        frequencies, times, spectrogram = signal.spectrogram(samples, self.config.data.sampleRate, nperseg=512,
                                                             noverlap=274)

        # Remove bad examples
        if spectrogram.shape != (257, 200):
            return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(
                np.random.rand(1, 257, 200)), torch.LongTensor([2])

        # Audio
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        audio = torch.Tensor(spectrogram.reshape(spec_shape))
        ret_dict = {'image': image, 'audio': audio, 'target': torch.LongTensor(result), 'vidClasses': vidClasses}
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
        self.fps = self.config.data.fps
        self.time = self.config.data.v_time
        # the origin code sample all the frames in one video, i.e. self.frame_sample_s = 1
        self.frame_sample_s = self.config.data.frame_sample_s
        # Frames per video
        self.fpv = 1 + self.fps * (self.time - 1) // self.frame_sample_s
        # total frames of all the videos
        self.tot_frames = len(self.vid2genre) * self.fpv
        self.length = self.tot_frames

        self._vid_transform = self.get_video_transform()
        self._aud_transform = self.get_audio_transform

    def get_video_transform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_list.append(vtransforms.Resize(self.config.data.imgSize, Image.BICUBIC))
        transform_list.append(vtransforms.CenterCrop(self.config.data.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
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

    def __getitem__(self, idx):
        result = [0]
        video_idx = int(idx / self.fpv)
        video_frame_number = idx % self.fpv
        frame_time = 500 + (video_frame_number * 1000 * self.frame_sample_s / self.fps)
        rate, samples = wav.read(os.path.join(self.audio_path, self.audio_files[video_idx] + '.wav'))
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
            ret_dict = {'image': torch.Tensor(np.random.rand(3, 224, 224)),
                        'audio': torch.Tensor(np.random.rand(1, 257, 200)),
                        'target': torch.LongTensor([2]), 'vidClasses': vidClasses}
            return ret_dict
        ##############################
        # Bring the channel to front
        image = image.transpose(2, 0, 1)
        image = self._vid_transform(torch.Tensor(image))
        # select 1 sec
        start = int(time * 48000) - 24000
        end = int(time * 48000) + 24000
        samples = samples[start:end]
        samples = self._aud_transform(samples)
        frequencies, times, spectrogram = signal.spectrogram(samples, self.config.data.sampleRate, nperseg=512,
                                                             noverlap=274)

        # Remove bad examples
        if spectrogram.shape != (257, 200):
            return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(
                np.random.rand(1, 257, 200)), torch.LongTensor([2])

        # Audio
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        audio = torch.Tensor(spectrogram.reshape(spec_shape))
        ret_dict = {'image': image, 'audio': audio, 'target': torch.LongTensor(result), 'vidClasses': vidClasses}
        return ret_dict
