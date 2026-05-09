import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}
 
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False, use_adaptive_pool=True):
        super(VGG, self).__init__()
        self.features = features
        self.use_adaptive_pool = use_adaptive_pool
        
        if use_adaptive_pool:
            # 使用自适应池化，支持任意输入尺寸
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes)
            )
        else:
            # 原始固定尺寸方式
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x C x H x W
        x = self.features(x)
        # N x 512 x H' x W'
        if self.use_adaptive_pool:
            x = self.avgpool(x)  # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list, input_channels=3):
    layers = []
    in_channels = input_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

# vgg_tiny(VGG11), vgg_small(VGG13), vgg(VGG16), vgg_big(VGG19)
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],   
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=1000, input_channels=3, use_adaptive_pool=True): 
    cfg = cfgs["vgg11"]
    model = VGG(make_features(cfg, input_channels=input_channels), num_classes=num_classes, use_adaptive_pool=use_adaptive_pool)
    return model

def vgg13(num_classes=1000, input_channels=3, use_adaptive_pool=True):  
    cfg = cfgs["vgg13"]
    model = VGG(make_features(cfg, input_channels=input_channels), num_classes=num_classes, use_adaptive_pool=use_adaptive_pool)
    return model

def vgg16(num_classes=1000, input_channels=3, use_adaptive_pool=True):  
    cfg = cfgs["vgg16"]
    model = VGG(make_features(cfg, input_channels=input_channels), num_classes=num_classes, use_adaptive_pool=use_adaptive_pool)
    return model

def vgg19(num_classes=1000, input_channels=3, use_adaptive_pool=True):  
    cfg = cfgs['vgg19']
    model = VGG(make_features(cfg, input_channels=input_channels), num_classes=num_classes, use_adaptive_pool=use_adaptive_pool)
    return model