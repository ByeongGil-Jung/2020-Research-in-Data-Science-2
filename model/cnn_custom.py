from torch import nn
from torch.nn import functional as F

from model.base import ModelBase


class CNNCustom(ModelBase):

    def __init__(self, fc_input):
        super(CNNCustom, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 224),
            nn.LeakyReLU(inplace=True),
            nn.Linear(224, 1)
        )

        self.init_weight()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(x.size()[0], -1)

        x = self.fc(x)
        x = F.sigmoid(x)

        return x
