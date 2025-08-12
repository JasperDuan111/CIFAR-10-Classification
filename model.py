from xml.parsers.expat import model
from torch import nn
import torch

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x
    

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.BatchNorm2d(reduce_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 使用两层3x3卷积来代替5x5卷积
        # 这样可以减少参数量和计算量，同时保持感受野
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.BatchNorm2d(reduce_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionNet(nn.Module):
    def __init__(self): 
        super(InceptionNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), 
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第一次下采样：32x32 -> 16x16
        )
        
        # 输出通道数计算方法： out_1x1 + out_3x3 + out_5x5 + pool_proj
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)  # 输出: 64+128+32+32 = 256
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)  # 输出: 128+192+96+64 = 480 

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第二次下采样：16x16 -> 8x8

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)  # 输出: 192+208+48+64 = 512
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)  # 输出: 160+224+64+64 = 512

        # 为CIFAR-10简化网络，移除部分层
        # self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)  # 输出: 128+256+64+64 = 512
        # self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)  # 输出: 112+288+64+64 = 528
        # self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)  # 输出: 256+320+128+128 = 832 

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第三次下采样：8x8 -> 4x4

        self.inception5a = InceptionBlock(512, 256, 160, 320, 32, 128, 128)  # 输出: 256+320+128+128 = 832
        # self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)  # 输出: 384+384+128+128 = 1024 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(832),
            nn.Dropout(0.3),
            nn.Linear(832, 10)  
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        # x = self.inception4c(x)  # 简化网络
        # x = self.inception4d(x)
        # x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        # x = self.inception5b(x)  # 简化网络
        x = self.avgpool(x)
        x = self.fc(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

       # 如果输入和输出通道不匹配，添加一个1x1卷积来调整维度
        self.shortcut = nn.Sequential()
        # 当 stride != 1 (需要下采样) 或通道数发生变化时，使用 1x1 卷积匹配尺寸
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 添加shortcut连接
        out += self.shortcut(identity)
        out = self.relu(out)

        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
    

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # 特征提取层 - 针对CIFAR-10 (32x32)进行调整
        self.features = nn.Sequential(
            # 第一层：卷积+ReLU+池化
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x64
            
            # 第二层：卷积+ReLU+池化
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 16x16x192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x192
            
            # 第三层：卷积+ReLU
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 8x8x384
            nn.ReLU(inplace=True),
            
            # 第四层：卷积+ReLU
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 8x8x256
            nn.ReLU(inplace=True),
            
            # 第五层：卷积+ReLU+池化
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x256
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 2x2x256
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 1024),  # 1024 = 256 * 2 * 2
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class VGGNet(nn.Module):
    def __init__(self, num_classes=10, vgg_type='VGG16'):
        super(VGGNet, self).__init__()
        
        # VGG配置：每个数字表示卷积层的输出通道数，'M'表示MaxPool
        configs = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        
        self.features = self._make_layers(configs[vgg_type])
        
        # 自适应池化，确保输出固定尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def _make_layers(self, config):
        layers = []
        in_channels = 3
        
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
    

# 测试模型是否适用于CIFAR-10

def test_mynet_model():
    model = Mynet()
    model.eval()

    # 创建一个CIFAR-10尺寸的测试输入 (batch_size=4, channels=3, height=32, width=32)
    test_input = torch.randn(4, 3, 32, 32)
    print("Input shape:", test_input.shape)

    with torch.no_grad():
        output = model(test_input)

    print("Output shape:", output.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("CIFAR-10 Mynet model test passed!")
    print("=" * 60)

    return model

def test_inception_model():
    model = InceptionNet()
    model.eval()

    # 创建一个CIFAR-10尺寸的测试输入 (batch_size=4, channels=3, height=32, width=32)
    test_input = torch.randn(4, 3, 32, 32)
    print("Input shape:", test_input.shape)
    
    with torch.no_grad():
        output = model(test_input)

    print("Output shape:", output.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("CIFAR-10 Inception model test passed!")
    print("=" * 60)
    return model

def test_ResNet_model():
    model = ResNet18()
    model.eval()

    # 创建一个CIFAR-10尺寸的测试输入 (batch_size=4, channels=3, height=32, width=32)
    test_input = torch.randn(4, 3, 32, 32)
    print("Input shape:", test_input.shape)
    
    with torch.no_grad():
        output = model(test_input)

    print("Output shape:", output.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("CIFAR-10 ResNet model test passed!")
    print("=" * 60)
    return model

def test_alexnet_model():
    model = AlexNet()
    model.eval()

    # 创建一个CIFAR-10尺寸的测试输入 (batch_size=4, channels=3, height=32, width=32)
    test_input = torch.randn(4, 3, 32, 32)
    print("Input shape:", test_input.shape)
    
    with torch.no_grad():
        output = model(test_input)

    print("Output shape:", output.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("CIFAR-10 AlexNet model test passed!")
    print("=" * 60)
    return model


def test_vggnet_model():
    # 测试不同的VGG变体
    vgg_types = ['VGG11', 'VGG13', 'VGG16', 'VGG19']
    
    for vgg_type in vgg_types:
        print(f"\nTesting {vgg_type}:")
        model = VGGNet(vgg_type=vgg_type)
        model.eval()

        # 创建一个CIFAR-10尺寸的测试输入 (batch_size=4, channels=3, height=32, width=32)
        test_input = torch.randn(4, 3, 32, 32)
        print("Input shape:", test_input.shape)
        
        with torch.no_grad():
            output = model(test_input)

        print("Output shape:", output.shape)
        print("Model parameters:", sum(p.numel() for p in model.parameters()))
        print(f"CIFAR-10 {vgg_type} model test passed!")
    print("=" * 60)
    return model


if __name__ == "__main__":
    test_mynet_model()
    test_inception_model()
    test_ResNet_model()
    test_alexnet_model()

    test_vggnet_model()
