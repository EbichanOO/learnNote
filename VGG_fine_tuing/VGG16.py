import chainer.links as L
import chainer.functions as F
from chainer import Chain

class VGG16(Chain):
    def __init__(self):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.fc6 = L.Linear(512*7*7, )

    def __call__(self, x):
        h = self.conv1_1(x)
        h = F.max_pooling_2d(self.conv1_2(h), 2)
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pooling_2d(self.conv3_1(h), 2)
        return h


import numpy as np
conver = VGG16()
x = np.random.rand(5, 3, 224, 224).astype(np.float32)
h = conver(x)
print(h.data.shape)