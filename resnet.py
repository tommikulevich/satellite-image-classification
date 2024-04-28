import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = ConvBlock(in_channels, out_channels, stride) \
            if in_channels != out_channels or stride != 1 else nn.ReLU()
    
    def forward(self, x):
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        
        return x
    
class FinalBlock(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.linear(torch.flatten(x, 1))
        
        return x

class ResidualNetwork(nn.Module):
    def __init__(self, layers, num_classes):
        super().__init__()
        self.initial_channels = 64
        self.channels = [self.initial_channels, 128, 256, 512]
        
        self.conv = ConvBlock(3, self.initial_channels)
        
        self.in_channels = self.initial_channels
        self.res_combination1 = self._make_layer(self.channels[0], layers[0])
        self.res_combination2 = self._make_layer(self.channels[1], layers[1], 2)
        self.res_combination3 = self._make_layer(self.channels[2], layers[2], 2)
        self.res_combination4 = self._make_layer(self.channels[3], layers[3], 2)
        
        self.final = FinalBlock(self.channels[3], num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)] + \
            [ResidualBlock(out_channels, out_channels) for _ in range(1, blocks)]
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        
        x = self.res_combination1(x)
        x = self.res_combination2(x)
        x = self.res_combination3(x)
        x = self.res_combination4(x)
        
        x = self.final(x)
        
        return x