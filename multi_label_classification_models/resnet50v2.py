import torch
import keras
from keras import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Add, ReLU, Input


@keras.saving.register_keras_serializable()
class ResnetV2Bottleneck(Model):
    """
    Replace basic blocks with Bottleneck blocks (using three convolutions per block: 1x1, 3x3, 1x1).
    """
    def __init__(self, out_channels, stride=1, downsample=False):
        super().__init__()
        mid_channels = out_channels // 4
        self.downsample = downsample
        self.stride = stride

        # Pre-activation bottleneck
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(mid_channels, (1, 1), strides=1, padding='same', kernel_initializer="he_normal")
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(mid_channels, (3, 3), strides=stride, padding='same', kernel_initializer="he_normal")
        self.bn3 = BatchNormalization()
        self.conv3 = Conv2D(out_channels, (1, 1), strides=1, padding='same', kernel_initializer="he_normal")

        if downsample or stride != 1:
            self.proj = Conv2D(out_channels, (1, 1), strides=stride, padding='same', kernel_initializer="he_normal")
            self.proj_bn = BatchNormalization()

        self.add = Add()

    def call(self, x):
        shortcut = x

        x = self.bn1(x)
        x = ReLU()(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = ReLU()(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = ReLU()(x)
        x = self.conv3(x)

        if self.downsample or self.stride != 1:
            shortcut = self.proj_bn(self.proj(shortcut))

        x = self.add([x, shortcut])
        return x

    def build(self, input_shape):
        # Create a dummy input tensor with the correct shape (excluding batch dim)
        dummy_input_shape = input_shape[1:]
        dummy_input = torch.randn((1, *dummy_input_shape))
        self.call(dummy_input)
        self.built = True


@keras.saving.register_keras_serializable()
class ResNet50V2(Model):
    def __init__(self, num_classes, use_case='multi-label', **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.use_case = use_case

        self.conv1 = Conv2D(64, (7,7), strides=2, padding='same', kernel_initializer="he_normal")
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.pool = MaxPool2D((3,3), strides=2, padding='same')

        # 3, 4, 6, 3 bottleneck blocks with expansion=4
        self.layer1 = self.make_layer(256, 3, stride=1)    # Stage 1: 64*4=256
        self.layer2 = self.make_layer(512, 4, stride=2)    # Stage 2: 128*4=512
        self.layer3 = self.make_layer(1024, 6, stride=2)   # Stage 3: 256*4=1024
        self.layer4 = self.make_layer(2048, 3, stride=2)   # Stage 4: 512*4=2048

        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.avgpool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax" if use_case=='multi-class' else "sigmoid")

    @staticmethod
    def make_layer(out_channels, blocks, stride=1):
        layers = []
        layers.append(ResnetV2Bottleneck(out_channels, stride=stride, downsample=True))
        for _ in range(1, blocks):
            layers.append(ResnetV2Bottleneck(out_channels, stride=1))
        return layers

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


    def build(self, input_shape):
        # Create a dummy input tensor with the correct shape (excluding batch dim)
        dummy_input_shape = input_shape[1:]
        dummy_input = torch.randn((1, *dummy_input_shape))
        self.call(dummy_input)
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'use_case': self.use_case,
        })
        return config
