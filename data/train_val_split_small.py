import os
import re
import numpy as np
import shutil
import json


def get_vid2genre(vid2genre_path):
    vid2genre = {}
    with open(vid2genre_path, 'r') as fin:
        lines = [line for line in fin.readlines() if not line.startswith('#')]
    for i, line in enumerate(lines):
        # Extract the words consisting of video_id, start_time, end_time, list of video_tags
        words = [word.replace("\n", "").replace('"', '') for word in line.replace(" ", "").split(",")]
        words = words[0:3] + [words[3:]]
        video_id = words[0]
        vid2genre[video_id] = words[-1]
    return vid2genre


if __name__ == '__main__':
    # A script for splitting the small dataset, 100 pair
    vid2genre_path = './data_infos/videos.csv'
    vid2genre = get_vid2genre(vid2genre_path)

    # get all video id
    data_mall_dir = './data_small'
    video_ids = [i for i in os.listdir(os.path.join(data_mall_dir, 'Video'))]
    data_ids = []
    for v_name in video_ids:
        match_object = re.match("video_(.*).mp4", v_name)
        data_ids.append(match_object.group(1))

    # split train and val
    np.random.shuffle(data_ids)
    train_split_ind = int(0.8 * len(data_ids))
    print('num of train data:%d, num of val data:%d' % (train_split_ind, len(data_ids) - train_split_ind))
    train_data_ids = data_ids[:train_split_ind]
    val_data_ids = data_ids[train_split_ind:]

    # deal with train data_ids
    train_vid2genre = {}
    train_data_dir = os.path.join(data_mall_dir, 'train')
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    video_train_data_dir = os.path.join(train_data_dir, 'video')
    if not os.path.exists(video_train_data_dir):
        os.makedirs(video_train_data_dir)
    audio_train_data_dir = os.path.join(train_data_dir, 'audio')
    if not os.path.exists(audio_train_data_dir):
        os.makedirs(audio_train_data_dir)
    for v_name in train_data_ids:
        train_vid2genre[v_name] = vid2genre[v_name]
        shutil.copy(src='./data_small/Video/video_%s.mp4' % v_name,
                    dst=os.path.join(video_train_data_dir, '%s.mp4' % v_name))
        shutil.copy(src='./data_small/Audio/audio_%s.wav' % v_name,
                    dst=os.path.join(audio_train_data_dir, '%s.wav' % v_name))
    with open(os.path.join(train_data_dir, 'vid2genre.json'), 'w') as fin:
        json.dump(train_vid2genre, fin)

    # deal with val data_ids
    val_vid2genre = {}
    val_data_dir = os.path.join(data_mall_dir, 'val')
    if not os.path.exists(val_data_dir):
        os.makedirs(val_data_dir)
    video_val_data_dir = os.path.join(val_data_dir, 'video')
    if not os.path.exists(video_val_data_dir):
        os.makedirs(video_val_data_dir)
    audio_val_data_dir = os.path.join(val_data_dir, 'audio')
    if not os.path.exists(audio_val_data_dir):
        os.makedirs(audio_val_data_dir)
    for v_name in val_data_ids:
        val_vid2genre[v_name] = vid2genre[v_name]
        shutil.copy(src='./data_small/Video/video_%s.mp4' % v_name,
                    dst=os.path.join(video_val_data_dir, '%s.mp4' % v_name))
        shutil.copy(src='./data_small/Audio/audio_%s.wav' % v_name,
                    dst=os.path.join(audio_val_data_dir, '%s.wav' % v_name))
    with open(os.path.join(val_data_dir, 'vid2genre.json'), 'w') as fin:
        json.dump(val_vid2genre, fin)
