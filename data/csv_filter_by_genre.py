import json
import numpy as np
import os
import shutil

if __name__ == '__main__':
    # genre which you want
    select_genre_file = './data_infos/genre_select_paper.txt'
    with open(select_genre_file, 'r') as fin:
        select_genre_list = [line.replace('\n', '') for line in fin.readlines()]
    print('total number of class:%d' % len(select_genre_list))
    # genre to video index(id)
    genre2tag_file = './data_infos/genre_tag_map.json'
    with open(genre2tag_file, 'r') as fin:
        genre_tag_dict = json.load(fin)
    tag_genre_dict = {val: key for key, val in genre_tag_dict.items()}
    select_tag_list = []
    for name in select_genre_list:
        select_tag_list.append(tag_genre_dict[name])

    # filter train csv file
    train_csv_path = "./data_infos/unbalanced_train_segments.csv"
    train_csv_filter_path = "./data_infos/unbalanced_train_segments_filtered.csv"
    with open(train_csv_path, 'r') as fin:
        lines = [line for line in fin.readlines() if not line.startswith('#')]
    train_lines_filtered = []
    for i, line in enumerate(lines):
        words = [word.replace("\n", "").replace('"', '') for word in line.replace(" ", "").split(",")]
        words = words[0:3] + [words[3:]]
        video_id = words[0]
        new_tags = set(select_tag_list) & set(words[-1])
        if len(new_tags) > 0:
            words_new = words[0:3] + [words[-1]]
            train_lines_filtered.append(words_new)
    print('total number of train data after filtering:%d' % len(train_lines_filtered))
    with open(train_csv_filter_path, 'w') as fw:
        for i, words_new in enumerate(train_lines_filtered):
            line_new = words_new[:3] + ['"' + ','.join(words_new[-1]) + '"']
            line_new = ','.join(line_new) + '\n'
            fw.write(line_new)

    # filter validation csv file
    val_csv_path = "./data_infos/eval_segments.csv"
    val_csv_filter_path = "./data_infos/eval_segments_filtered.csv"
    with open(val_csv_path, 'r') as fin:
        lines = [line for line in fin.readlines() if not line.startswith('#')]
    val_lines_filtered = []
    for i, line in enumerate(lines):
        words = [word.replace("\n", "").replace('"', '') for word in line.replace(" ", "").split(",")]
        words = words[0:3] + [words[3:]]
        video_id = words[0]
        new_tags = set(select_tag_list) & set(words[-1])
        if len(new_tags) > 0:
            words_new = words[0:3] + [words[-1]]
            val_lines_filtered.append(words_new)
    print('total number of val data after filtering:%d' % len(val_lines_filtered))
    with open(val_csv_filter_path, 'w') as fw:
        for i, words_new in enumerate(val_lines_filtered):
            line_new = words_new[:3] + ['"' + ','.join(words_new[-1]) + '"']
            line_new = ','.join(line_new) + '\n'
            fw.write(line_new)
