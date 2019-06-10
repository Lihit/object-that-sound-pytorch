import os
import os.path as osp
from easydict import EasyDict as edict

config = edict()
config.version = 1
config.exp_description = "baseline of AVENet, version%d" % config.version
config.gpus = "0"
config.store_dir = "/data2/wenshaoguo/object_that_sound"
config.store_name = "AVENet_AudioSet_baseline_v%d" % config.version

# add data config
config.data = edict()
config.data.train_data_dir = "./data/data_small/train"
config.data.val_data_dir = "./data/data_small/val"
config.data.test_data_dir = "./data/data_small/test"
config.data.fps = 25
config.data.v_time = 10
config.data.frame_sample_s = 25
config.data.imgSize = 224
config.data.audRate = 48000
config.data.audLen = 48000
config.data.batch_size = 64
config.data.stft_frame = 512
config.data.stft_hop = 241
config.data.AudH = 257
config.data.AudW = 200

# add train config
config.train = edict()
config.train.pretrained_model = ""
config.train.num_workers = 16
config.train.lr = 0.001
config.train.weight_decay = 5e-04
config.train.epochs = 200
config.train.val_epoch = 10
config.train.model_save_epoch = 10
config.train.is_train = True

# add visualize config
config.vis = edict()
config.vis.env = "AVENet_AudioSet_baseline_v%d" % config.version
config.vis.DEFAULT_HOSTNAME = "http://localhost"
config.vis.DEFAULT_PORT = 8097
