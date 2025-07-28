"""
https://github.com/songrise/CNN_Keras/blob/main/src/ResNet-18.py
ResNet-18
Reference:
[1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In
ICCV, 2015.
"""
import keras
import torch

from typing import Literal
from keras import Model
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Add, ReLU


@keras.saving.register_keras_serializable()
class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = ReLU()(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = ReLU()(x)
        return out

    def build(self, input_shape):
        # Create a dummy input tensor with the correct shape (excluding batch dim)
        dummy_input_shape = input_shape[1:]
        dummy_input = torch.randn((1, *dummy_input_shape))
        self.call(dummy_input)
        self.built = True

    def compute_output_shape(self, input_shape):
        # Compute how shape changes due to convolutions + strides
        batch, channels, height, width = input_shape
        new_height = height // 2 if self.__down_sample else height
        new_width = width // 2 if self.__down_sample else width
        new_channels = self.__channels
        return batch, new_channels, new_height, new_width


@keras.saving.register_keras_serializable()
class ResNet18(Model):

    def __init__(self, num_classes, use_case: Literal['multi-class', 'multi-label'] = 'multi-label', **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.use_case = use_case
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax" if use_case == 'multi-class' else "sigmoid")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = ReLU()(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

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
