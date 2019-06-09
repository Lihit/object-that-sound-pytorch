##  说明
主要参考代码：https://github.com/rohitrango/objects-that-sound 和 https://github.com/hangzhaomit/Sound-of-Pixels

论文：![objects-that-sound](https://arxiv.org/pdf/1712.06651.pdf) 和 ![Sound-of-Pixels](https://arxiv.org/abs/1804.03160)
### 数据准备
* 下载数据，该论文使用的是AudioSet这个数据集，但是这个数据集只提供youtube的video id，需要自己去下载，在`./data/`目录里我提供了下载
的脚本`download_video_audio.py`,但是需要梯子（你懂的）。建议先将所有的数据下载，然后再根据自己的需求划分，训练集videoid文件对应是
`./data/data_infos/balanced_train_segments.csv`和`./data/data_infos/unbalanced_train_segments.csv`, 二者的区别是类别均不均衡，
其中`unbalanced_train_segments.csv`包含的数据量比较多，根据自己的需求选择吧。验证集videoid文件对应是
`./data/data_infos/eval_segments.csv`。然后更改`download_video_audio.py`里的文件路径，运行`python download_video_audio.py`.
* 根据自己的需求选择数据。`./data/select_data_by_genre.py`就是根据类别`./data/data_infos/genre_select.txt`来挑选数据，可以参考这个
脚本写自己的挑选规则。
* 脚本`./data/train_val_split_small.py`是对鑫哥给我的100对数据进行划分和处理。

### 数据格式
```
data   
│
└───train
│   │       vid2genre.json
│   └───video
│       │   0FaO2iU5q2s.mp4
│       │   0FbD8rmclVE.mp4
│       │   ...
│   └───audio
│       │   0FaO2iU5q2s.wav
│       │   0FbD8rmclVE.wav
│       │   ...
│   
└───val
│   │       vid2genre.json
│   └───video
│       │   0FaO2iU5q2s.mp4
│       │   ...
│   └───audio
│       │   0FaO2iU5q2s.wav
│       │   ...
└───test
│   │      vid2genre.json
│   └───video
│       │   0FaO2iU5q2s.mp4
│       │   ...
│   └───audio
│       │   0FaO2iU5q2s.wav
│       │   ...
```

每个数据目录下应该包含`train, val, test`三个文件目录，用于训练，验证和测试。每个子目录，比如train下，都有一个文件`vid2genre.json`，
里面每个video id对应他的类别，注意在Youtube的video类别是编码之后的，可通过`./data/data_infos/class_labels_indices.csv`解码。另外还有
`audio`和`video`两个文件夹，分别用于保存视频和音频，注意这两个目录下的每个数据的id都要做到互相对应（correspond).如果要跑自己的数据集的话，
请按照这样的数据格式处理，不然你可能需要在`./dataset`里自己写dataset和dataloader。

### 运行环境
* python3
* GPU
* python的一些包：pytorch, numpy, pandas, opencv, scipy等，缺什么再补上。

### 运行
* 修改`configs/config_AVENet.py`里一些配置，比如数据路径等，如果你新建了一个config文件，记得在`train_AVENet.py`
调用他。
* 然后运行:`python train_AVENet.py`

### TODO
* `train_VOLNet.py`待写
* 论文里nDCG的评价指标待写，这个得定义每个类别之间的关系，进而定义视频直接的联系。
* 贴上一些结果图
