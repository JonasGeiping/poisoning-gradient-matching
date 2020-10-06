"""Model definitions."""

import torch
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck

from collections import OrderedDict

from .mobilenet import MobileNetV2
from .vgg import VGG


def get_model(model_name, dataset_name, pretrained=False):
    """Retrieve an appropriate architecture."""
    if 'CIFAR' in dataset_name or 'MNIST' in dataset_name:
        if pretrained:
            raise ValueError('Loading pretrained models is only supported for ImageNet.')
        in_channels = 1 if dataset_name == 'MNIST' else 3
        num_classes = 10 if dataset_name in ['CIFAR10', 'MNIST'] else 100
        if 'ResNet' in model_name:
            model = resnet_picker(model_name, dataset_name)
        elif 'efficientnet-b' in model_name.lower():
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_name(model_name.lower())
        elif model_name == 'ConvNet':
            model = convnet(width=32, in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'ConvNet64':
            model = convnet(width=64, in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'ConvNet128':
            model = convnet(width=64, in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'ConvNetBN':
            model = ConvNetBN(width=64, in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'Linear':
            model = linear_model(dataset_name, num_classes=num_classes)
        elif model_name == 'alexnet-mp':
            model = alexnet_metapoison(in_channels=in_channels, num_classes=num_classes, batchnorm=False)
        elif model_name == 'alexnet-mp-bn':
            model = alexnet_metapoison(in_channels=in_channels, num_classes=num_classes, batchnorm=True)
        elif 'VGG' in model_name:
            model = VGG(model_name)
        elif model_name == 'MobileNetV2':
            model = MobileNetV2(num_classes=num_classes, train_dp=0, test_dp=0, droplayer=0, bdp=0)
        else:
            raise ValueError(f'Architecture {model_name} not implemented for dataset {dataset_name}.')

    elif 'TinyImageNet' in dataset_name:
        in_channels = 3
        num_classes = 200

        if 'VGG16' in model_name:
            model = VGG('VGG16-TI', in_channels=in_channels, num_classes=num_classes)
        elif 'ResNet' in model_name:
            model = resnet_picker(model_name, dataset_name)
        else:
            raise ValueError(f'Model {model_name} not implemented for TinyImageNet')

    elif 'ImageNet' in dataset_name:
        in_channels = 3
        num_classes = 1000
        if 'efficientnet-b' in model_name.lower():
            from efficientnet_pytorch import EfficientNet
            if pretrained:
                model = EfficientNet.from_pretrained(model_name.lower())
            else:
                model = EfficientNet.from_name(model_name.lower())
        elif model_name == 'Linear':
            model = linear_model(dataset_name, num_classes=num_classes)
        else:
            if 'densenet' in model_name.lower():
                extra_args = dict(memory_efficient=False)  # memory_efficient->checkpointing -> incompatible with autograd.grad
            else:
                extra_args = dict()

            try:
                model = getattr(torchvision.models, model_name.lower())(pretrained=pretrained, **extra_args)
            except AttributeError:
                raise NotImplementedError(f'ImageNet model {model_name} not found at torchvision.models.')

    return model


def linear_model(dataset, num_classes=10):
    """Define the simplest linear model."""
    if 'cifar' in dataset.lower():
        dimension = 3072
    elif 'mnist' in dataset.lower():
        dimension = 784
    elif 'imagenet' in dataset.lower():
        dimension = 150528
    elif 'tinyimagenet' in dataset.lower():
        dimension = 64**2 * 3
    else:
        raise ValueError('Linear model not defined for dataset.')
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(dimension, num_classes))

def convnet(width=32, in_channels=3, num_classes=10, **kwargs):
    """Define a simple ConvNet. This architecture only really works for CIFAR10."""
    model = torch.nn.Sequential(OrderedDict([
        ('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)),
        ('relu0', torch.nn.ReLU()),
        ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu1', torch.nn.ReLU()),
        ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu2', torch.nn.ReLU()),
        ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu3', torch.nn.ReLU()),
        ('pool3', torch.nn.MaxPool2d(3)),
        ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu4', torch.nn.ReLU()),
        ('pool4', torch.nn.MaxPool2d(3)),
        ('flatten', torch.nn.Flatten()),
        ('linear', torch.nn.Linear(36 * width, num_classes))
    ]))
    return model


def alexnet_metapoison(widths=[16, 32, 32, 64, 64], in_channels=3, num_classes=10, batchnorm=False):
    """AlexNet variant as used in MetaPoison."""
    def convblock(width_in, width_out):
        if batchnorm:
            bn = torch.nn.BatchNorm2d(width_out)
        else:
            bn = torch.nn.Identity()
        return torch.nn.Sequential(torch.nn.Conv2d(width_in, width_out, kernel_size=3, padding=1),
                                   torch.nn.ReLU(),
                                   bn,
                                   torch.nn.MaxPool2d(2, 2))
    blocks = []
    width_in = in_channels
    for width in widths:
        blocks.append(convblock(width_in, width))
        width_in = width

    model = torch.nn.Sequential(*blocks, torch.nn.Flatten(), torch.nn.Linear(widths[-1], num_classes))
    return model


class ConvNetBN(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=64, num_classes=10, in_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),

            ('flatten', torch.nn.Flatten()),
            ('linear', torch.nn.Linear(36 * width, num_classes))
        ]))

    def forward(self, input):
        return self.model(input)


def resnet_picker(arch, dataset):
    """Pick an appropriate resnet architecture for MNIST/CIFAR."""
    in_channels = 1 if dataset == 'MNIST' else 3
    num_classes = 10
    if dataset in ['CIFAR10', 'MNIST']:
        num_classes = 10
        initial_conv = [3, 1, 1]
    elif dataset == 'CIFAR100':
        num_classes = 100
        initial_conv = [3, 1, 1]
    elif dataset == 'TinyImageNet':
        num_classes = 200
        initial_conv = [7, 2, 3]
    else:
        raise ValueError(f'Unknown dataset {dataset} for ResNet.')

    if arch == 'ResNet20':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif 'ResNet20-' in arch and arch[-1].isdigit():
        width_factor = int(arch[-1])
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * width_factor, initial_conv=initial_conv)
    elif arch == 'ResNet28-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [4, 4, 4], num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet32':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet32-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet44':
        return ResNet(torchvision.models.resnet.BasicBlock, [7, 7, 7], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet56':
        return ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet110':
        return ResNet(torchvision.models.resnet.BasicBlock, [18, 18, 18], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet18':
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif 'ResNet18-' in arch:  # this breaks the usual notation, but is nicer for now!!
        new_width = int(arch.split('-')[1])
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=new_width, initial_conv=initial_conv)
    elif arch == 'ResNet34':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet50':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet101':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet152':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    else:
        raise ValueError(f'Invalid ResNet [{dataset}] model chosen: {arch}.')


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR-like thingies.

    This is a minor modification of
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
    adding additional options.
    """

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=[False, False, False, False],
                 norm_layer=torch.nn.BatchNorm2d, strides=[1, 2, 2, 2], initial_conv=[3, 1, 1]):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # torch.nn.Module
        self._norm_layer = norm_layer

        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=initial_conv[0],
                                     stride=initial_conv[1], padding=initial_conv[2], bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)

        layer_list = []
        width = self.inplanes
        for idx, layer in enumerate(layers):
            layer_list.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2
        self.layers = torch.nn.Sequential(*layer_list)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the arch by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
