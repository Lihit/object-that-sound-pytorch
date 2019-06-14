# coding: utf-8
import json
import youtube_dl
import subprocess
import os
import shutil
import multiprocessing
import argparse


def download_vid(line, train_data_dir):
    # Extract the words consisting of video_id, start_time, end_time, list of video_tags
    words = [word.replace("\n", "").replace('"', '') for word in line.replace(" ", "").split(",")]
    words = words[0:3] + [words[3:]]
    video_id = words[0]

    train_data_dir_full = os.path.join(train_data_dir, 'full')

    ydl_opts = {
        'start_time': int(float(words[1])),
        'end_time': int(float(words[2])),
        'format': 'mp4[height<=360]',
        'outtmpl': r"{}/{}.%(ext)s".format(train_data_dir_full, video_id)
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=' + video_id])

        info = ydl.extract_info(str('https://www.youtube.com/watch?v=' + video_id),
                                download=False)  # Extract Info without download
        ext = info.get('ext', None)  # Extract the extension of the downloaded video
        full_video_path = os.path.join(train_data_dir_full, video_id + "." + ext)
        video_file_path = os.path.join(train_data_dir, 'video', video_id + ".mp4")
        if not os.path.exists(os.path.dirname(video_file_path)):
            os.makedirs(os.path.dirname(video_file_path))
        # -r mean fps=25
        subprocess.call(
            ["ffmpeg", "-ss", str(int(float(words[1]))), "-i", full_video_path, "-r", "25", "-t", "00:00:10",
             "-vcodec", "copy", "-acodec", "copy", video_file_path])
        audio_file_path = os.path.join(train_data_dir, 'audio', video_id + ".wav")
        if not os.path.exists(os.path.dirname(audio_file_path)):
            os.makedirs(os.path.dirname(audio_file_path))
        command = ["ffmpeg", "-i", video_file_path, "-ab", "160k", "-ac", "1", "-ar", "48000", "-vn",
                   audio_file_path]
        subprocess.call(command)
    print("Im Done")


def downloadAllVideos(train_csv_path, train_data_dir):
    """
    download all the video by csv file
    :param train_csv_path: csv file path, which contain download data info
    :param train_data_dir: dir to save the data
    :return: vid2genre: dict of video:genre
    """

    vid2genre = {}
    with open(train_csv_path, 'r') as fin:
        lines = [line for line in fin.readlines() if not line.startswith('#')]
    print('start downloading %d videos' % len(lines))
    # use multiprocessing pool
    pool = multiprocessing.Pool(16)
    for i, line in enumerate(lines):
        # Extract the words consisting of video_id, start_time, end_time, list of video_tags
        words = [word.replace("\n", "").replace('"', '') for word in line.replace(" ", "").split(",")]
        words = words[0:3] + [words[3:]]
        video_id = words[0]
        vid2genre[video_id] = words[-1]
        pool.apply_async(download_vid, (line, train_data_dir))

    pool.close()
    pool.join()
    return vid2genre


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data_all',
                        help="data dir which you want to save")
    parser.add_argument('--train_csv_path', default='./data_infos/unbalanced_train_segments_filtered.csv',
                        help="train csv file which you want to download")
    parser.add_argument('--val_csv_path', default='./data_infos/eval_segments_filtered.csv',
                        help="val csv file which you want to download")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="number worker to download")
    args = parser.parse_args()
    # download all video first, without any preprocessing
    # path to save all the data
    all_data_dir = args.data_dir
    if not os.path.exists(all_data_dir):
        os.makedirs(all_data_dir)

    # download the train data
    train_csv_path = args.train_csv_path
    train_data_dir = os.path.join(all_data_dir, 'train')
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    train_data_dir_full = os.path.join(train_data_dir, 'full')
    if not os.path.exists(train_data_dir_full):
        os.makedirs(train_data_dir_full)
    print('start downloading the train data:%s to %s' % (train_csv_path, train_data_dir))
    train_vid2genre = downloadAllVideos(train_csv_path, train_data_dir)
    shutil.rmtree(train_data_dir_full)
    with open(os.path.join(train_data_dir, 'vid2genre.json'), 'w') as fin:
        json.dump(train_vid2genre, fin)

    # download the val data
    val_csv_path = args.val_csv_path
    val_data_dir = os.path.join(all_data_dir, 'val')
    if not os.path.exists(val_data_dir):
        os.makedirs(val_data_dir)
    val_data_dir_full = os.path.join(val_data_dir, 'full')
    if not os.path.exists(val_data_dir_full):
        os.makedirs(val_data_dir_full)
    print('start downloading the val data:%s to %s' % (val_csv_path, val_data_dir))
    val_vid2genre = downloadAllVideos(val_csv_path, val_data_dir)
    shutil.rmtree(val_data_dir_full)
    with open(os.path.join(val_data_dir, 'vid2genre.json'), 'w') as fin:
        json.dump(val_vid2genre, fin)
