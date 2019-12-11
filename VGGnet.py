#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class VGGNet(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as input
    """

    def __init__(self):
        super(VGGNet, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )


    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8(h)

        return h


class VGGNetsmall(chainer.Chain):
    """
    VGGNet
    - It takes (224, 224, 3) sized image as input
    """

    def __init__(self):
        super(VGGNetsmall, self).__init__(
            conv1_1=L.Convolution2D(3, 16, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(16, 16, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(16, 32, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(32, 32, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(32, 64, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(64, 32, 3, stride=1, pad=1),

            fc6=L.Linear(25088, 1024),
            fc7=L.Linear(1024, 128),
            fc8=L.Linear(128, 2)
        )

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))

        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), ratio=0.3)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.3)
        h = self.fc8(h)

        return h

class VGGNetsmall2(chainer.Chain):
            """
            VGGNet
            - It takes (224, 224, 3) sized image as input
            """

            def __init__(self):
                super(VGGNetsmall2, self).__init__(
                    conv1_1=L.Convolution2D(3, 8, 3, stride=1, pad=1),
                    conv1_2=L.Convolution2D(8, 8, 3, stride=1, pad=1),

                    conv2_1=L.Convolution2D(8, 16, 3, stride=1, pad=1),
                    conv2_2=L.Convolution2D(16, 16, 3, stride=1, pad=1),

                    conv3_1=L.Convolution2D(16, 32, 3, stride=1, pad=1),
                    conv3_2=L.Convolution2D(32, 32, 3, stride=1, pad=1),
                    conv3_3=L.Convolution2D(32, 16, 3, stride=1, pad=1),

                    fc6=L.Linear(12544, 1024),
                    fc7=L.Linear(1024, 128),
                    fc8=L.Linear(128, 2)
                )

            def __call__(self, x):
                h = F.relu(self.conv1_1(x))

                h = F.relu(self.conv1_2(h))
                h = F.max_pooling_2d(h, 2, stride=2)

                h = F.relu(self.conv2_1(h))
                h = F.relu(self.conv2_2(h))
                h = F.max_pooling_2d(h, 2, stride=2)

                h = F.relu(self.conv3_1(h))
                h = F.relu(self.conv3_2(h))
                h = F.relu(self.conv3_3(h))
                h = F.max_pooling_2d(h, 2, stride=2)

                # h = F.relu(self.conv4_1(h))
                # h = F.relu(self.conv4_2(h))
                # h = F.relu(self.conv4_3(h))
                # h = F.max_pooling_2d(h, 2, stride=2)

                h = F.dropout(F.relu(self.fc6(h)), ratio=0.3)
                h = F.dropout(F.relu(self.fc7(h)), ratio=0.3)
                h = self.fc8(h)

                return h