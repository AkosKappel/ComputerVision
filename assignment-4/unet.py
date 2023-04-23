from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


# ConvBlock = [Conv2D -> BatchNorm -> Activation -> Dropout] * n_blocks
def conv_block(
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        padding: str = 'same',
        kernel_initializer: str = 'he_normal',
        kernel_regularizer: str = None,
        batchnorm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.0,
        n_blocks: int = 2,
):
    x = input_tensor
    for i in range(1, n_blocks + 1):
        x = layers.Conv2D(
            filters=n_filters, kernel_size=(kernel_size, kernel_size), padding=padding,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
        )(input_tensor)
        if batchnorm:
            x = layers.BatchNormalization(axis=-1)(x)
        if activation:
            x = layers.Activation(activation)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    return x


def encoder_block(input_tensor, n_filters: int, kernel_size: int = 3, **kwargs):
    x = conv_block(input_tensor, n_filters=n_filters, kernel_size=kernel_size, **kwargs)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(input_tensor, skip_tensors, n_filters: int, kernel_size: int = 3, **kwargs):
    u = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    c = layers.concatenate([u, *skip_tensors])
    x = conv_block(c, n_filters=n_filters, kernel_size=kernel_size, **kwargs)
    return x


def UNet(input_shape, num_classes: int = 1, num_filters: int = 32, final_activation='sigmoid', **kwargs):
    """
    input -> pool1, conv1 -----------------------------------------> (+) up4 -> output
               |                                                      ^
               v                                                      |
             pool1 -> pool2, conv2 ----------------------------> (+) up3
                        |                                         ^
                        v                                         |
                      pool2 -> pool3, conv3 ---------------> (+) up2
                                 |                            ^
                                 v                            |
                               pool3 -> pool4, conv4 --> (+) up1
                                          |               ^
                                          v               |
                                        pool4    ->    bridge
    """
    filters = [num_filters * 2 ** i for i in range(5)]  # [32, 64, 128, 256, 512]
    inputs = layers.Input(shape=input_shape)

    # Contracting path (encoder)
    conv1, pool1 = encoder_block(inputs, filters[0], kernel_size=3, **kwargs)
    conv2, pool2 = encoder_block(pool1, filters[1], kernel_size=3, **kwargs)
    conv3, pool3 = encoder_block(pool2, filters[2], kernel_size=3, **kwargs)
    conv4, pool4 = encoder_block(pool3, filters[3], kernel_size=3, **kwargs)

    # Bottleneck (bridge)
    conv5 = conv_block(pool4, filters[4], kernel_size=3, **kwargs)

    # Expansive path (decoder)
    up1 = decoder_block(conv5, [conv4], filters[3], kernel_size=3, **kwargs)
    up2 = decoder_block(up1, [conv3], filters[2], kernel_size=3, **kwargs)
    up3 = decoder_block(up2, [conv2], filters[1], kernel_size=3, **kwargs)
    up4 = decoder_block(up3, [conv1], filters[0], kernel_size=3, **kwargs)

    outputs = conv_block(
        up4, num_classes, kernel_size=1, n_blocks=1, activation=final_activation,
        batchnorm=False, dropout=0, kernel_regularizer=regularizers.l2(1e-4),
    )

    model = keras.Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    return model


# U-Net++ resources:
# https://github.com/CarryHJR/Nested-UNet/blob/master/model.py
# https://www.kaggle.com/code/ekhtiar/tf-tutorial-semantic-segmentation-with-u-net/notebook

def UNetPlusPlus(input_shape, num_classes: int = 1, num_filters: int = 32, deep_supervision=False,
                 final_activation='sigmoid', **kwargs):
    filters = [num_filters * 2 ** i for i in range(5)]
    inputs = layers.Input(shape=input_shape)

    # Backbone
    conv1_1, pool1 = encoder_block(inputs, filters[0], kernel_size=3, **kwargs)
    conv2_1, pool2 = encoder_block(pool1, filters[1], kernel_size=3, **kwargs)
    conv3_1, pool3 = encoder_block(pool2, filters[2], kernel_size=3, **kwargs)
    conv4_1, pool4 = encoder_block(pool3, filters[3], kernel_size=3, **kwargs)

    # Bottleneck
    conv5_1 = conv_block(pool4, filters[4], kernel_size=3, **kwargs)

    # Dense connections 1
    conv1_2 = decoder_block(conv2_1, [conv1_1], filters[0], kernel_size=3, **kwargs)
    conv2_2 = decoder_block(conv3_1, [conv2_1], filters[1], kernel_size=3, **kwargs)
    conv3_2 = decoder_block(conv4_1, [conv3_1], filters[2], kernel_size=3, **kwargs)
    conv4_2 = decoder_block(conv5_1, [conv4_1], filters[3], kernel_size=3, **kwargs)

    # Dense connections 2
    conv1_3 = decoder_block(conv2_2, [conv1_1, conv1_2], filters[0], kernel_size=3, **kwargs)
    conv2_3 = decoder_block(conv3_2, [conv2_1, conv2_2], filters[1], kernel_size=3, **kwargs)
    conv3_3 = decoder_block(conv4_2, [conv3_1, conv3_2], filters[2], kernel_size=3, **kwargs)

    # Dense connections 3
    conv1_4 = decoder_block(conv2_3, [conv1_1, conv1_2, conv1_3], filters[0], kernel_size=3, **kwargs)
    conv2_4 = decoder_block(conv3_3, [conv2_1, conv2_2, conv2_3], filters[1], kernel_size=3, **kwargs)

    # Dense connections 4 (final block of the outermost U-Net)
    conv1_5 = decoder_block(conv2_4, [conv1_1, conv1_2, conv1_3, conv1_4], filters[0], kernel_size=3, **kwargs)

    # Outputs
    nested_output1 = conv_block(
        conv1_2, num_classes, kernel_size=1, n_blocks=1, activation=final_activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )
    nested_output2 = conv_block(
        conv1_3, num_classes, kernel_size=1, n_blocks=1, activation=final_activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )
    nested_output3 = conv_block(
        conv1_4, num_classes, kernel_size=1, n_blocks=1, activation=final_activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )
    nested_output4 = conv_block(
        conv1_5, num_classes, kernel_size=1, n_blocks=1, activation=final_activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )

    # Deep supervision
    if deep_supervision:
        outputs = [nested_output1, nested_output2, nested_output3, nested_output4]
    else:
        outputs = [nested_output4]

    model = keras.Model(inputs=[inputs], outputs=outputs, name='U-NetPlusPlus')
    return model


def UNet3Plus(input_shape, num_classes: int = 1, num_filters: int = 32, deep_supervision=False,
              final_activation='sigmoid', **kwargs):
    filters = [num_filters * 2 ** i for i in range(5)]
    inputs = layers.Input(shape=input_shape)

    # Encoder blocks
    conv1, pool1 = encoder_block(inputs, filters[0], kernel_size=3, **kwargs)
    conv2, pool2 = encoder_block(pool1, filters[1], kernel_size=3, **kwargs)
    conv3, pool3 = encoder_block(pool2, filters[2], kernel_size=3, **kwargs)
    conv4, pool4 = encoder_block(pool3, filters[3], kernel_size=3, **kwargs)
    conv5 = conv_block(pool4, filters[4], kernel_size=3, **kwargs)

    # Decoder blocks
    n_concat_channels = filters[0]
    n_upsample_channels = len(filters) * n_concat_channels

    # Up 4
    conv1_up4 = layers.MaxPool2D((8, 8))(conv1)  # 320*320*64  --> 40*40*64
    conv1_up4 = conv_block(conv1_up4, n_concat_channels, kernel_size=3, n_blocks=1)  # 320*320*64  --> 40*40*64

    conv2_up4 = layers.MaxPool2D((4, 4))(conv2)  # 160*160*128 --> 40*40*128
    conv2_up4 = conv_block(conv2_up4, n_concat_channels, kernel_size=3, n_blocks=1)  # 160*160*128 --> 40*40*64

    conv3_up4 = layers.MaxPool2D((2, 2))(conv3)  # 80*80*256  --> 40*40*256
    conv3_up4 = conv_block(conv3_up4, n_concat_channels, kernel_size=3, n_blocks=1)  # 80*80*256  --> 40*40*64

    conv4_up4 = conv_block(conv4, n_concat_channels, kernel_size=3, n_blocks=1)  # 40*40*512  --> 40*40*64

    conv5_up4 = layers.UpSampling2D((2, 2), interpolation='bilinear')(conv5)  # 80*80*256  --> 40*40*256
    conv5_up4 = conv_block(conv5_up4, n_concat_channels, kernel_size=3, n_blocks=1)  # 20*20*1024  --> 20*20*64

    up4 = layers.concatenate([conv1_up4, conv2_up4, conv3_up4, conv4_up4, conv5_up4])
    up4 = conv_block(up4, n_upsample_channels, kernel_size=3, n_blocks=1)  # 40*40*320  --> 40*40*320

    # Up 3
    conv1_up3 = layers.MaxPool2D((4, 4))(conv1)  # 320*320*64 --> 80*80*64
    conv1_up3 = conv_block(conv1_up3, n_concat_channels, kernel_size=3, n_blocks=1)  # 80*80*64 --> 80*80*64

    conv2_up3 = layers.MaxPool2D((2, 2))(conv2)  # 160*160*256 --> 80*80*256
    conv2_up3 = conv_block(conv2_up3, n_concat_channels, kernel_size=3, n_blocks=1)  # 80*80*256 --> 80*80*64

    conv3_up3 = conv_block(conv3, n_concat_channels, kernel_size=3, n_blocks=1)  # 80*80*512 --> 80*80*64

    conv4_up3 = layers.UpSampling2D((2, 2), interpolation='bilinear')(conv4)  # 40*40*320 --> 80*80*320
    conv4_up3 = conv_block(conv4_up3, n_concat_channels, kernel_size=3, n_blocks=1)  # 80*80*320 --> 80*80*64

    conv5_up3 = layers.UpSampling2D((4, 4), interpolation='bilinear')(conv5)  # 20*20*320 --> 80*80*320
    conv5_up3 = conv_block(conv5_up3, n_concat_channels, kernel_size=3, n_blocks=1)  # 80*80*320 --> 80*80*64

    up3 = layers.concatenate([conv1_up3, conv2_up3, conv3_up3, conv4_up3, conv5_up3])
    up3 = conv_block(up3, n_upsample_channels, kernel_size=3, n_blocks=1)  # 80*80*320 --> 80*80*320

    # Up 2
    conv1_up2 = layers.MaxPool2D((2, 2))(conv1)  # 320*320*64 --> 160*160*64
    conv1_up2 = conv_block(conv1_up2, n_concat_channels, kernel_size=3, n_blocks=1)  # 160*160*64 --> 160*160*64

    conv2_up2 = conv_block(conv2, n_concat_channels, kernel_size=3, n_blocks=1)  # 160*160*256 --> 160*160*64

    conv3_up2 = layers.UpSampling2D((2, 2), interpolation='bilinear')(up3)  # 80*80*320 --> 160*160*320
    conv3_up2 = conv_block(conv3_up2, n_concat_channels, kernel_size=3, n_blocks=1)  # 160*160*320 --> 160*160*64

    conv4_up2 = layers.UpSampling2D((4, 4), interpolation='bilinear')(up4)  # 40*40*320 --> 160*160*320
    conv4_up2 = conv_block(conv4_up2, n_concat_channels, kernel_size=3, n_blocks=1)  # 160*160*320 --> 160*160*64

    conv5_up2 = layers.UpSampling2D((8, 8), interpolation='bilinear')(conv5)  # 20*20*320 --> 160*160*320
    conv5_up2 = conv_block(conv5_up2, n_concat_channels, kernel_size=3, n_blocks=1)  # 160*160*320 --> 160*160*64

    up2 = layers.concatenate([conv1_up2, conv2_up2, conv3_up2, conv4_up2, conv5_up2])
    up2 = conv_block(up2, n_upsample_channels, kernel_size=3, n_blocks=1)  # 160*160*320 --> 160*160*320

    # Up 1
    conv1_up1 = conv_block(conv1, n_concat_channels, kernel_size=3, n_blocks=1)  # 320*320*64 --> 320*320*64

    conv2_up1 = layers.UpSampling2D((2, 2), interpolation='bilinear')(up2)  # 160*160*320 --> 320*320*320
    conv2_up1 = conv_block(conv2_up1, n_concat_channels, n_blocks=1)  # 160*160*320 --> 160*160*64

    conv3_up1 = layers.UpSampling2D((4, 4), interpolation='bilinear')(up3)  # 80*80*320 --> 320*320*320
    conv3_up1 = conv_block(conv3_up1, n_concat_channels, n_blocks=1)  # 320*320*320 --> 320*320*64

    conv4_up1 = layers.UpSampling2D((8, 8), interpolation='bilinear')(up4)  # 40*40*320 --> 320*320*320
    conv4_up1 = conv_block(conv4_up1, n_concat_channels, n_blocks=1)  # 320*320*320 --> 320*320*64

    conv5_up1 = layers.UpSampling2D((16, 16), interpolation='bilinear')(conv5)  # 20*20*320 --> 320*320*320
    conv5_up1 = conv_block(conv5_up1, n_concat_channels, n_blocks=1)  # 320*320*320 --> 320*320*64

    up1 = layers.concatenate([conv1_up1, conv2_up1, conv3_up1, conv4_up1, conv5_up1])
    up1 = conv_block(up1, n_upsample_channels, kernel_size=3, n_blocks=1)  # 320*320*320 --> 320*320*320

    # Output
    output1 = conv_block(up1, num_classes, kernel_size=3, n_blocks=1,
                         batchnorm=False, activation='', dropout=0)  # 320*320*320 --> 320*320*1
    output2 = conv_block(up2, num_classes, kernel_size=3, n_blocks=1,
                         batchnorm=False, activation='', dropout=0)  # 160*160*320 --> 160*160*1
    output3 = conv_block(up3, num_classes, kernel_size=3, n_blocks=1,
                         batchnorm=False, activation='', dropout=0)  # 80*80*320 --> 80*80*1
    output4 = conv_block(up4, num_classes, kernel_size=3, n_blocks=1,
                         batchnorm=False, activation='', dropout=0)  # 40*40*320 --> 40*40*1
    output5 = conv_block(conv5, num_classes, kernel_size=3, n_blocks=1,
                         batchnorm=False, activation='', dropout=0)  # 20*20*320 --> 20*20*1

    # Upsample
    # No need to upsample output1
    output2 = layers.UpSampling2D((2, 2), interpolation='bilinear')(output2)  # 160*160*1 --> 320*320*1
    output3 = layers.UpSampling2D((4, 4), interpolation='bilinear')(output3)  # 80*80*1 --> 320*320*1
    output4 = layers.UpSampling2D((8, 8), interpolation='bilinear')(output4)  # 40*40*1 --> 320*320*1
    output5 = layers.UpSampling2D((16, 16), interpolation='bilinear')(output5)  # 20*20*1 --> 320*320*1

    # Activations
    output1 = layers.Activation(final_activation)(output1)
    output2 = layers.Activation(final_activation)(output2)
    output3 = layers.Activation(final_activation)(output3)
    output4 = layers.Activation(final_activation)(output4)
    output5 = layers.Activation(final_activation)(output5)

    # Deep Supervision
    if deep_supervision:
        outputs = [output5, output4, output3, output2, output1]
    else:
        outputs = [output1]

    model = keras.Model(inputs=inputs, outputs=outputs, name='UNet3Plus')
    return model


# Residual Skip Block
def RSB(input_tensor, kernel_size, filters, strides=1, skip=True):
    f1, f2, f3, f4 = filters
    # First component of main path
    x = layers.Conv2D(f1, (1, 1), strides=strides)(input_tensor)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Activation('relu')(x)
    # Second component of main path
    x = layers.Conv2D(f2, kernel_size, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Activation('relu')(x)
    # Third component of main path
    x = layers.Conv2D(f3, (1, 1))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    # Shortcut path
    if skip:
        shortcut = input_tensor
    else:
        shortcut = layers.Conv2D(f4, (1, 1), strides=strides)(input_tensor)
        shortcut = layers.BatchNormalization(momentum=0.8)(shortcut)
    # Add shortcut value to main path, and pass it through a RELU activation
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def SuimEncoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    # First encoder block
    x = layers.Conv2D(64, (5, 5), strides=1)(inputs)
    encoded1 = x
    # Second encoder block
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    x = RSB(x, kernel_size=3, filters=[64, 64, 128, 128], strides=2, skip=False)
    x = RSB(x, kernel_size=3, filters=[64, 64, 128, 128], strides=1, skip=True)
    x = RSB(x, kernel_size=3, filters=[64, 64, 128, 128], strides=1, skip=True)
    encoded2 = x
    # Third encoder block
    x = RSB(x, kernel_size=3, filters=[128, 128, 256, 256], strides=2, skip=False)
    x = RSB(x, kernel_size=3, filters=[128, 128, 256, 256], strides=1, skip=True)
    x = RSB(x, kernel_size=3, filters=[128, 128, 256, 256], strides=1, skip=True)
    x = RSB(x, kernel_size=3, filters=[128, 128, 256, 256], strides=1, skip=True)
    encoded3 = x
    # Encoder output
    return inputs, [encoded1, encoded2, encoded3]


def SuimDecoder(encoded_inputs, num_classes):
    def concat_skip(layer_input, skip_input, filters, f_size=3):
        u = layers.Conv2D(filters, (f_size, f_size), strides=1, padding='same', activation='relu')(layer_input)
        u = layers.BatchNormalization(momentum=0.8)(u)
        u = layers.Concatenate()([u, skip_input])
        return u

    # First decoder block
    encoded1, encoded2, encoded3 = encoded_inputs
    decoded1 = layers.Conv2D(256, (3, 3), padding='same')(encoded3)
    decoded1 = layers.BatchNormalization(momentum=0.8)(decoded1)
    decoded1 = layers.UpSampling2D((2, 2))(decoded1)
    # Padding to match dimensions
    decoded1 = layers.Lambda(lambda x: x[:, :-2, :-2, :])(decoded1)
    decoded1 = layers.ZeroPadding2D((1, 1))(decoded1)
    encoded2 = layers.Lambda(lambda x: x[:, :-1, :-1, :])(encoded2)
    encoded2 = layers.ZeroPadding2D((1, 1))(encoded2)
    decoded1_skip = concat_skip(encoded2, decoded1, 256)
    # Second decoder block
    decoded2 = layers.Conv2D(256, (3, 3), strides=1, padding='same')(decoded1_skip)
    decoded2 = layers.BatchNormalization(momentum=0.8)(decoded2)
    decoded2 = layers.UpSampling2D((2, 2))(decoded2)
    decoded2_skip = layers.Conv2D(128, (3, 3), strides=1, padding='same')(decoded2)
    decoded2_skip = layers.BatchNormalization(momentum=0.8)(decoded2_skip)
    decoded2_skip = layers.UpSampling2D((2, 2))(decoded2_skip)
    # Padding to match dimensions
    encoded1 = layers.ZeroPadding2D((2, 2))(encoded1)
    decoded2_skip = concat_skip(encoded1, decoded2_skip, 128)
    # Third decoder block
    decoded3 = layers.Conv2D(128, (3, 3), strides=1, padding='same')(decoded2_skip)
    decoded3 = layers.BatchNormalization()(decoded3)
    decoded3_skip = layers.Conv2D(64, (3, 3), strides=1, padding='same')(decoded3)
    decoded3_skip = layers.BatchNormalization(momentum=0.8)(decoded3_skip)
    # Decoder output
    output = layers.Conv2D(num_classes, (3, 3), strides=1, padding='same', activation='sigmoid')(decoded3_skip)
    return output


def SuimNet(input_shape, num_classes):
    inputs, encoded_inputs = SuimEncoder(input_shape)
    output = SuimDecoder(encoded_inputs, num_classes)
    model = keras.Model(input=inputs, output=output, name='SuimNet')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model
