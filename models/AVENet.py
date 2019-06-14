from .image_convnet import *
from .audio_convnet import *
import torch
from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class AVENet(nn.Module):
    """
    AVENET
    """

    def __init__(self):
        super(AVENet, self).__init__()

        self.relu = F.relu
        self.imgnet = ImageConvNet()
        self.audnet = AudioConvNet()
        self.flatten = Flatten()

        # Vision subnetwork
        # add by wenshao, maybe the globalaveragepool/globalmaxpool will be flexible?
        self.vpool4 = nn.AdaptiveMaxPool2d(1)
        self.vfc1 = nn.Linear(512, 128)
        self.vfc2 = nn.Linear(128, 128)
        # modify by wenshao, nn.BatchNorm1d is learnable layer, which is different from the paper, should be l2norm
        self.vl2norm = self.l2norm

        # Audio subnetwork
        # add by wenshao, maybe the globalaveragepool/globalmaxpool will be flexible?
        self.apool4 = nn.AdaptiveMaxPool2d(1)
        self.afc1 = nn.Linear(512, 128)
        self.afc2 = nn.Linear(128, 128)
        self.al2norm = self.l2norm

        # Combining layers
        self.eucdis = self.EucDistance
        self.fc3 = nn.Linear(1, 2)

        # need to initialize the tiny fc
        self.fc3.weight.data[0] = -0.7090
        self.fc3.weight.data[1] = 0.7090
        self.fc3.bias.data[0] = 1.2186
        self.fc3.bias.data[1] = - 1.2186

    def l2norm(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-08
        x = torch.div(x, norm)
        return x

    def EucDistance(self, x1, x2):
        d = (x1 - x2).pow(2).sum(dim=1, keepdim=True).sqrt()
        return d

    def forward(self, image, audio):
        # Image
        img = self.imgnet(image)
        img = self.vpool4(img)
        img = self.flatten(img)
        img = self.relu(self.vfc1(img))
        img = self.vfc2(img)
        img = self.vl2norm(img)

        # Audio
        aud = self.audnet(audio)
        aud = self.apool4(aud)
        aud = self.flatten(aud)
        aud = self.relu(self.afc1(aud))
        aud = self.afc2(aud)
        aud = self.al2norm(aud)

        # Join them
        mse = self.eucdis(img, aud)
        out = self.fc3(mse)

        return out, img, aud

    def get_image_embeddings(self, image):
        # Just get the image embeddings
        img = self.imgnet(image)
        img = self.vpool4(img)
        img = self.flatten(img)
        img = self.relu(self.vfc1(img))
        img = self.vfc2(img)
        img = self.vl2norm(img)
        return img

    def get_audio_embeddings(self, audio):
        # Just get the audio embeddings
        aud = self.audnet(audio)
        aud = self.apool4(aud)
        aud = self.flatten(aud)
        aud = self.relu(self.afc1(aud))
        aud = self.afc2(aud)
        aud = self.al2norm(aud)
        return aud
