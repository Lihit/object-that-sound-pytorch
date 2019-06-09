from .image_convnet import *
from .audio_convnet import *
import torch
from torch import nn
import torch.nn.functional as F


class AVOLNet(nn.Module):
    """
    AVOLNet, add object location compared with AVENet
    """

    def __init__(self):
        super(AVOLNet, self).__init__()

        self.relu = F.relu
        self.imgnet = ImageConvNet()
        self.audnet = AudioConvNet()

        # Vision subnetwork
        self.fconv1 = nn.Conv2d(512, 128, 1, 1, 0)
        self.fconv2 = nn.Conv2d(128, 129, 1, 1, 0)

        # Audio subnetwork
        # add by wenshao, maybe the globalaveragepool/globalmaxpool will be flexible?
        self.apool4 = nn.AdaptiveMaxPool2d(1)
        self.afc1 = nn.Linear(512, 128)
        self.afc2 = nn.Linear(128, 128)

        # Combining layers
        self.conv_merge = nn.Sequential(nn.Conv2d(1, 1, 1, 1, 0), nn.Sigmoid())
        self.maxout = nn.AdaptiveMaxPool2d(1)

    def forward(self, image, audio):
        # Image
        img = self.imgnet(image)
        img = self.fconv1(img)
        img = self.fconv2(img)

        # Audio
        aud = self.audnet(audio)
        aud = self.apool4(aud).squeeze(2).squeeze(2)
        aud = self.relu(self.afc1(aud))
        aud = self.afc2(aud)

        # Join them
        aud_r = aud.view(aud.size(0), 1, -1)
        img_r = img.view(img.size(0), 128, -1)
        img_aud_pro = torch.bmm(aud_r, img_r).view(aud.size(0), 1, 14, 14)
        loc = self.conv_merge(img_aud_pro)
        out = self.maxout(loc)

        return out, loc
