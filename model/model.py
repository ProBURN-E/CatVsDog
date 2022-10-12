import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

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

