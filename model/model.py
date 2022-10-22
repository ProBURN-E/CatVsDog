import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, mobilenet_v2

# todo: How to calculate the number of parameters in the model? trainable and non-trainable


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VGG16Model(BaseModel):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.sf = nn.Softmax(dim=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.fc4096_1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc4096_2 = nn.Linear(4096, 4096)
        self.fc_end = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1(x)
        x = self.conv1_2(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.relu1(x)
        x = self.conv2_2(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv3_1(x)
        x = self.relu1(x)
        x = self.conv3_2(x)
        x = self.relu1(x)
        x = self.conv3_3(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv4_1(x)
        x = self.relu1(x)
        x = self.conv4_2(x)
        x = self.relu1(x)
        x = self.conv4_3(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv5_1(x)
        x = self.relu1(x)
        x = self.conv5_2(x)
        x = self.relu1(x)
        x = self.conv5_3(x)
        x = self.relu1(x)

        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc4096_1(x)
        x = self.relu1(x)
        x = self.fc4096_2(x)
        x = self.relu1(x)
        x = self.fc_end(x)
        x = self.sf(x)

        return x


class Resnet18Model(BaseModel):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet18(pretrained=True)
        fc_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fc_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class Resnet50Model(BaseModel):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet50(pretrained=True)
        fc_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fc_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class VGG16FromVisionModel(BaseModel):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.vgg = vgg16(pretrained=True)
        fc_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(fc_features, self.num_classes)

    def forward(self, x):
        x = self.vgg(x)
        return x


class BasicBlock(nn.Module):  # ResNet残差块
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1):
        super().__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        #         print('shape of x: {}'.format(x.shape))
        out = self.layer(x)
#         print('shape of out: {}'.format(out.shape))
#         print('After shortcut shape of x: {}'.format(self.shortcut(x).shape))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置


class ResNet18SelfModel(nn.Module):
    def __init__(self, BasicBlock=BasicBlock, num_classes=10):
        super().__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])
        # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])
        # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])
        # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    # 这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

#         out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

class MobileNetV2Model(BaseModel):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.mobilenet = mobilenet_v2(pretrained=True)
        fc_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(fc_features, self.num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        return x