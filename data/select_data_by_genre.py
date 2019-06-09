import json
import numpy as np
import os
import pandas as pd
import shutil


def select_data(all_data_dir, select_data_dir, select_tag_list):
    """
    select data
    :param all_data_dir: dir of all data
    :param select_data_dir: dir of select data, where you want to save
    :param select_tag_list: selected tags which you want
    :return: None
    """
    with open(os.path.join(all_data_dir, 'vid2genre.json'), 'r') as fin:
        vid2genre = json.load(fin)
    vid2genre_new = {}
    for key in vid2genre:
        new_tags = set(select_tag_list) & set(vid2genre[key])
        if len(new_tags):
            vid2genre_new[key] = list(new_tags)
    # copy video
    select_data_video_dir = os.path.join(select_data_dir, 'video')
    if not os.path.exists(select_data_video_dir):
        os.makedirs(select_data_video_dir)
    for video_key in vid2genre_new:
        shutil.copy(src=os.path.join(all_data_dir, 'video', '%s.mp4' % video_key),
                    dst=select_data_video_dir)

    # copy audio
    select_data_audio_dir = os.path.join(select_data_dir, 'audio')
    if not os.path.exists(select_data_audio_dir):
        os.makedirs(select_data_audio_dir)
    for video_key in vid2genre_new:
        shutil.copy(src=os.path.join(all_data_dir, 'audio', '%s.wav' % video_key),
                    dst=select_data_audio_dir)

    # save vid2genre_new
    with open(os.path.join(select_data_dir, 'vid2genre.json'), 'w') as fin:
        json.dump(vid2genre_new, fin)
    print('number of select data:%d' % len(vid2genre_new))


if __name__ == '__main__':
    select_data_dir = './data_select'
    if not os.path.exists(select_data_dir):
        os.makedirs(select_data_dir)
    # genre which you want
    select_genre_file = './data_infos/genre_select.txt'
    with open(select_genre_file, 'r') as fin:
        select_genre_list = [line.replace('\n', '') for line in fin.readlines()]
    # genre to video index(id)
    genre2tag_file = './data_infos/genre_tag_map.json'
    with open(genre2tag_file, 'r') as fin:
        genre_tag_dict = json.load(fin)
    tag_genre_dict = {val: key for key, val in genre_tag_dict.items()}
    select_tag_list = []
    for name in select_genre_list:
        select_tag_list.append(tag_genre_dict[name])

    # select data in the all train data
    print('start selecting train data')
    all_data_dir = './data_all'
    train_data_dir = os.path.join(all_data_dir, 'train')
    select_train_data_dir = os.path.join(select_data_dir, 'train')
    select_data(train_data_dir, select_train_data_dir, select_tag_list)

    # select data in the all val data
    all_data_dir = './data_all'
    val_data_dir = os.path.join(all_data_dir, 'val')
    select_val_data_dir = os.path.join(select_data_dir, 'val')
    select_data(val_data_dir, select_val_data_dir, select_tag_list)
    print('start selecting val data')
