from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers


# U-Net++ resources:
# https://github.com/CarryHJR/Nested-UNet/blob/master/model.py
# https://www.kaggle.com/code/ekhtiar/tf-tutorial-semantic-segmentation-with-u-net/notebook

def dice_coef(y_true, y_pred, smooth: float = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred, smooth: float = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def jaccard_loss(y_true, y_pred):
    return 1.0 - jaccard_coef(y_true, y_pred)


def standard_block(
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        batchnorm: bool = True,
        kernel_initializer: str = 'he_normal',
        kernel_regularizer: str = None,
        padding: str = 'same',
        activation: str = 'relu',
        dropout: float = 0.1,
):  # standard unit = Conv2D, BatchNorm, Activation, Dropout
    x = layers.Conv2D(
        filters=n_filters, kernel_size=(kernel_size, kernel_size), padding=padding,
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
    )(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation(activation)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def conv2d_block(
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        batchnorm: bool = True,
        kernel_initializer: str = 'he_normal',
        kernel_regularizer: str = None,
        padding: str = 'same',
        activation: str = 'relu',
        dropout: float = 0.1,
):
    # First convolutional layer
    x = standard_block(
        input_tensor, n_filters=n_filters, kernel_size=kernel_size, batchnorm=batchnorm, activation=activation,
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, padding=padding, dropout=dropout
    )
    # Second convolutional layer
    x = standard_block(
        x, n_filters=n_filters, kernel_size=kernel_size, batchnorm=batchnorm, activation=activation,
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, padding=padding, dropout=dropout,
    )
    return x


def encoder_block(input_tensor, n_filters: int, kernel_size: int = 3, **kwargs):
    x = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=kernel_size, **kwargs)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(input_tensor, skip_tensors, n_filters: int, kernel_size: int = 3, **kwargs):
    u = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    c = layers.concatenate([u, *skip_tensors])
    x = conv2d_block(c, n_filters=n_filters, kernel_size=kernel_size, **kwargs)
    return x


def UNet(input_shape, num_classes: int = 1, num_filters: int = 32, **kwargs):
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
                                          |              ^
                                          v              |
                                        pool4    ->    bridge
    """
    filters = [num_filters * 2 ** i for i in range(5)]
    inputs = layers.Input(shape=input_shape)

    # Contracting path (encoder)
    conv1, pool1 = encoder_block(inputs, filters[0], kernel_size=3, **kwargs)
    conv2, pool2 = encoder_block(pool1, filters[1], kernel_size=3, **kwargs)
    conv3, pool3 = encoder_block(pool2, filters[2], kernel_size=3, **kwargs)
    conv4, pool4 = encoder_block(pool3, filters[3], kernel_size=3, **kwargs)

    # Bottleneck (bridge)
    conv5 = conv2d_block(pool4, filters[4], kernel_size=3, **kwargs)

    # Expansive path (decoder)
    up1 = decoder_block(conv5, [conv4], filters[3], kernel_size=3, **kwargs)
    up2 = decoder_block(up1, [conv3], filters[2], kernel_size=3, **kwargs)
    up3 = decoder_block(up2, [conv2], filters[1], kernel_size=3, **kwargs)
    up4 = decoder_block(up3, [conv1], filters[0], kernel_size=3, **kwargs)

    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = standard_block(
        up4, num_classes, kernel_size=1, activation=activation,
        batchnorm=False, dropout=0, kernel_regularizer=regularizers.l2(1e-4),
    )

    model = keras.Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    return model


def UNetPlusPlus(input_shape, num_classes: int = 1, num_filters: int = 32, deep_supervision=True, **kwargs):
    filters = [num_filters * 2 ** i for i in range(5)]
    inputs = layers.Input(shape=input_shape)

    # Backbone
    conv1_1, pool1 = encoder_block(inputs, filters[0], kernel_size=3, **kwargs)
    conv2_1, pool2 = encoder_block(pool1, filters[1], kernel_size=3, **kwargs)
    conv3_1, pool3 = encoder_block(pool2, filters[2], kernel_size=3, **kwargs)
    conv4_1, pool4 = encoder_block(pool3, filters[3], kernel_size=3, **kwargs)

    # Bottleneck
    conv5_1 = conv2d_block(pool4, filters[4], kernel_size=3, **kwargs)

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
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    nested_output1 = standard_block(
        conv1_2, num_classes, kernel_size=1, activation=activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )
    nested_output2 = standard_block(
        conv1_3, num_classes, kernel_size=1, activation=activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )
    nested_output3 = standard_block(
        conv1_4, num_classes, kernel_size=1, activation=activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )
    nested_output4 = standard_block(
        conv1_5, num_classes, kernel_size=1, activation=activation,
        dropout=0, batchnorm=False, kernel_regularizer=regularizers.l2(1e-4),
    )

    # Deep supervision
    if deep_supervision:
        outputs = [nested_output1, nested_output2, nested_output3, nested_output4]
    else:
        outputs = [nested_output4]

    model = keras.Model(inputs=[inputs], outputs=[outputs], name='U-Net++')
    return model
