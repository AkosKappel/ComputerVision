import albumentations as A
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import wandb
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CLASS_NAMES = [
    'Background (waterbody)', 'Human divers', 'Aquatic plants and sea-grass',
    'Wrecks and ruins', 'Robots (AUVs/ROVs/instruments)', 'Reefs and invertebrates',
    'Fish and vertebrates', 'Sea-floor and rocks',
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_COLOR = {
    0: (0, 0, 0),  # Black
    1: (0, 0, 1),  # Blue
    2: (0, 1, 0),  # Green
    3: (0, 1, 1),  # Sky
    4: (1, 0, 0),  # Red
    5: (1, 0, 1),  # Pink
    6: (1, 1, 0),  # Yellow
    7: (1, 1, 1),  # White
}
COLOR_TO_CLASS = {v: k for k, v in CLASS_TO_COLOR.items()}
CLASS_COLORS = np.array([list(color) for color in CLASS_TO_COLOR.values()])
CLASS_WEIGHTS = np.array([1.06, 14.62, 15.72, 5.10, 44.18, 1.00, 4.78, 2.42])

_BATCH_SIZE = 4
_HEIGHT = 480
_WIDTH = 640
_CHANNELS = 3


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def list_files(path):
    content = os.listdir(path)
    # Get only files (not directories)
    files = [os.path.join(path, f) for f in content if os.path.isfile(os.path.join(path, f))]
    # Ensure the lists are sorted to match the corresponding image and mask
    files.sort()
    return files


def load_image(path):
    # Read RGB image
    image = cv.imread(path, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Resize
    image = cv.resize(image, (_WIDTH, _HEIGHT), interpolation=cv.INTER_LINEAR)
    # Normalize
    image = (image / 255.0).astype(np.float32)
    return image


def load_mask(path):
    # Read RGB mask
    mask = cv.imread(path, cv.IMREAD_COLOR)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    # Resize
    mask = cv.resize(mask, (_WIDTH, _HEIGHT), interpolation=cv.INTER_NEAREST)
    # Fix rounding errors
    mask = np.where(mask > 127, 1, 0)
    # Convert to one-hot encoding
    one_hot_mask = []
    for color in COLOR_TO_CLASS:
        mask_slice = np.all(np.equal(mask, color), axis=-1)
        one_hot_mask.append(mask_slice)
    one_hot_mask = np.stack(one_hot_mask, axis=-1).astype(np.float32)
    return one_hot_mask


# Encode RGB mask to class indices in range [0, num_classes - 1]
def rgb_to_class_indices(mask):
    class_colors_tensor = tf.constant(CLASS_COLORS, dtype=tf.float32)
    # Calculate the similarity between the RGB mask and class colors
    similarity = tf.reduce_sum(tf.abs(tf.subtract(mask, class_colors_tensor[:, None, None, :])), axis=-1)
    # Find the index of the class color with the minimum similarity
    class_indices_mask = tf.argmin(similarity, axis=0)
    return class_indices_mask


# One-hot encode class indices mask (e.g. [1, 3, 0] -> [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
def class_indices_to_one_hot(mask):
    one_hot_mask = tf.one_hot(mask, NUM_CLASSES, dtype=tf.int32)
    one_hot_mask = tf.cast(one_hot_mask, tf.float32)
    return one_hot_mask


# Encode RGB mask to one-hot mask (e.g. [[0, 0, 1], [1, 0, 0]] -> [[0, 1, 0, 0], [1, 0, 0, 0]])
def rgb_to_one_hot(mask):
    indices = rgb_to_class_indices(mask)
    one_hot_mask = class_indices_to_one_hot(indices)
    return one_hot_mask


# Decode one-hot mask to RGB mask
def class_indices_to_rgb(mask):
    class_colors_tensor = tf.constant(CLASS_COLORS, dtype=tf.float32)
    # Get the RGB color for each pixel from the class color
    rgb_mask = tf.gather(class_colors_tensor, tf.cast(mask, tf.int32))
    return rgb_mask


def one_hot_to_class_indices(mask):
    indices = tf.argmax(mask, axis=-1)
    return indices


def one_hot_to_rgb(mask):
    indices = one_hot_to_class_indices(mask)
    rgb_mask = class_indices_to_rgb(indices)
    return rgb_mask


def visualize_pair(image, mask, mask_type='rgb'):
    if mask_type == 'rgb':
        pass
    elif mask_type == 'one-hot':
        mask = one_hot_to_rgb(mask)
    elif mask_type == 'class-indices':
        mask = class_indices_to_rgb(mask)
    else:
        raise ValueError(f'Unknown mask type: {mask_type}')
    mask = mask.numpy().astype(np.uint8)
    mask *= 255

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(mask)
    plt.axis('off')
    plt.show()


Augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.75, interpolation=cv.INTER_NEAREST, border_mode=cv.BORDER_REFLECT_101),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ], p=0.75),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.MedianBlur(blur_limit=3, p=0.5),
    ], p=0.75),
])


# Augment with tensorflow
def augmentation(image, mask):
    augmented = Augmentation(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']
    return image, mask


def preprocessing(image, mask, augment=False, preprocess_input=None):
    def read(x, y):
        image = load_image(x.decode())
        mask = load_mask(y.decode())
        if preprocess_input is not None:
            image = preprocess_input(image)
        if augment:
            image, mask = augmentation(image, mask)
        return image, mask

    image, mask = tf.numpy_function(read, [image, mask], [tf.float32, tf.float32])
    image.set_shape([_HEIGHT, _WIDTH, _CHANNELS])
    mask.set_shape([_HEIGHT, _WIDTH, NUM_CLASSES])

    return image, mask


def load_dataset(path, val_split=0.2, seed=411,
                 batch_size=4, height=240, width=320, channels=3,
                 repeat=True, augment=True, preprocess=None):
    global _HEIGHT, _WIDTH, _CHANNELS, _BATCH_SIZE
    _HEIGHT, _WIDTH, _CHANNELS = height, width, channels
    _BATCH_SIZE = batch_size

    # Get file names
    train_val_images = list_files(os.path.join(path, 'train_val/images'))
    train_val_masks = list_files(os.path.join(path, 'train_val/masks'))
    test_images = list_files(os.path.join(path, 'TEST/images'))
    test_masks = list_files(os.path.join(path, 'TEST/masks'))

    # Split to train and validation
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=val_split, random_state=seed
    )

    # Create datasets from file name pairs (image, mask)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

    # Shuffle train dataset
    train_dataset = train_dataset.shuffle(len(train_images))

    # Preprocessing and augmentation
    preprocess_train = lambda x, y: preprocessing(x, y, augment=augment, preprocess_input=preprocess)
    preprocess_test = lambda x, y: preprocessing(x, y, augment=False, preprocess_input=preprocess)

    train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    train_dataset = train_dataset.batch(_BATCH_SIZE)
    val_dataset = val_dataset.batch(_BATCH_SIZE)
    test_dataset = test_dataset.batch(_BATCH_SIZE)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # Repeat train dataset indefinitely
    if repeat:
        train_dataset = train_dataset.repeat()

    # Return datasets
    return train_dataset, val_dataset, test_dataset


# ------------------------------------------------------------
# Loss functions & metrics
# ------------------------------------------------------------

def iou_index(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def f_score(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


# IoU = Jaccard index
def jaccard_coef(y_true, y_pred, smooth: float = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return K.mean((intersection + smooth) / (union + smooth))


# f1 score = Dice coefficient
def dice_coef(y_true, y_pred, smooth: float = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    area_true = K.sum(y_true_f)
    area_pred = K.sum(y_pred_f)
    return K.mean((2. * intersection + smooth) / (area_true + area_pred + smooth))


def jaccard_loss(y_true, y_pred):
    return 1.0 - jaccard_coef(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred, gamma=2., alpha=4., epsilon=1.e-9):
    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))

    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def hybrid_loss(y_true, y_pred):
    j_loss = jaccard_loss(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    f_loss = focal_loss(y_true, y_pred)
    return j_loss + d_loss + f_loss


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

class VisualizeProgress(tf.keras.callbacks.Callback):
    def __init__(self, test_data, every_n_epochs=10, num_samples=4,
                 convert_mask=None, deep_supervision=False, log=False):
        super(VisualizeProgress, self).__init__()
        self.test_data = test_data
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.convert_mask = convert_mask
        self.deep_supervision = deep_supervision
        self.log = log

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            x_test, y_test = self.test_data
            y_pred = self.model.predict(x_test[:self.num_samples])

            if self.deep_supervision:  # take the last prediction if deep supervision is used
                y_pred = y_pred[-1]

            plt.figure(figsize=(9, 12))
            for index in range(self.num_samples):
                if index >= len(x_test): break

                plt.subplot(self.num_samples, 3, index * 3 + 1)
                if index == 0:
                    plt.title('Input Image')
                plt.imshow(x_test[index])

                plt.subplot(self.num_samples, 3, index * 3 + 2)
                if index == 0:
                    plt.title('Ground Truth')
                ground_truth = self.convert_mask(y_test[index]).numpy()
                plt.imshow(ground_truth)

                plt.subplot(self.num_samples, 3, index * 3 + 3)
                if index == 0:
                    plt.title('Predicted Segmentation')
                prediction = self.convert_mask(y_pred[index]).numpy()
                plt.imshow(prediction)

            if self.log:
                wandb.log({f'Segmentation results': wandb.Image(plt)})

            plt.tight_layout()
            plt.show()


def plot_history(history):
    # Plot training history
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()

    plt.subplot(222)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()

    plt.subplot(223)
    plt.title('Jaccard')
    plt.plot(history.history['jaccard_coef'], label='train')
    plt.plot(history.history['val_jaccard_coef'], label='val')
    plt.legend()

    plt.subplot(224)
    plt.title('Dice')
    plt.plot(history.history['dice_coef'], label='train')
    plt.plot(history.history['val_dice_coef'], label='val')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training-history.png')
    plt.show()


# ------------------------------------------------------------
# Models & Building blocks
# ------------------------------------------------------------

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
    (+) means concatenation
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
    """                                      (Deep supervision)
                                  ----------------------------------------
                                  |            |            |            |
    input -> pool1, conv1_1 -> conv1_2  ->  conv1_3  ->  conv1_4  ->  conv1_5 -> output
               |                  ^            ^            ^            ^
               v                /            /            /            /
             pool1 -> pool2, conv2_1  ->  conv2_2  ->  conv2_3  ->  conv2_4
                        |                  ^            ^            ^
                        v                /            /            /
                      pool2 -> pool3, conv3_1  ->  conv3_2  ->  conv3_3
                                 |                  ^            ^
                                 v                /            /
                               pool3 -> pool4, conv4_1 -> conv4_2
                                          |                 ^
                                          v               /
                                        pool4   ->   conv5_1
    """
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
    """
    input -> pool1, conv1 -> ...                                 up1 -> output
               |                                                 ^
               v                                                /
             pool1 -> pool2, conv2 -> ...                     up2
                        |                                     ^
                        v                                    /
                      pool2 -> pool3, conv3 -> ...         up3
                                 |                         ^
                                 v                        /
                               pool3 -> pool4, conv4 -> up4
                                          |             ^
                                          v            /
                                        pool4   ->   conv5
    """
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


def gating_signal(input_tensor, n_filters, batchnorm=False):
    x = layers.Conv2D(n_filters, (1, 1), padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


# e.g. (None, 256, 256, 6) -> rep=2, axis=3 -> (None, 256, 256, 3)
def repeat_elem(tensor, rep, axis=-1):
    return layers.Lambda(lambda x, r: K.repeat_elements(x, r, axis=axis), arguments={'r': rep})(tensor)


def attention_block(input_tensor, gating_tensor, n_filters):
    input_shape = K.int_shape(input_tensor)
    gating_shape = K.int_shape(gating_tensor)

    # Change input signal to the same shape as the gating signal
    theta_input = layers.Conv2D(n_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    theta_input_shape = K.int_shape(theta_input)

    # Change the gating signal to the same number of filters as the input signal
    phi_gating = layers.Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same')(gating_tensor)
    sx = theta_input_shape[1] // gating_shape[1]
    sy = theta_input_shape[2] // gating_shape[2]
    up_gating = layers.Conv2DTranspose(n_filters, (3, 3), strides=(sx, sy), padding='same')(phi_gating)

    # Add the input and gating signals
    input_gating = layers.add([up_gating, theta_input])
    input_gating = layers.Activation('relu')(input_gating)
    input_gating = layers.Conv2D(1, (1, 1), padding='same')(input_gating)
    input_gating = layers.Activation('sigmoid')(input_gating)
    input_gating_shape = K.int_shape(input_gating)
    sx = input_shape[1] // input_gating_shape[1]
    sy = input_shape[2] // input_gating_shape[2]
    input_gating = layers.UpSampling2D((sx, sy))(input_gating)
    input_gating = repeat_elem(input_gating, input_shape[3])

    output = layers.multiply([input_gating, input_tensor])
    output = layers.Conv2D(input_shape[3], (1, 1), padding='same')(output)
    output = layers.BatchNormalization()(output)

    return output


def residual_conv_block(input_tensor, n_filters, kernel_size: int = 3, dropout: float = 0.0, batchnorm=False):
    conv = layers.Conv2D(n_filters, (kernel_size, kernel_size), padding='same')(input_tensor)
    if batchnorm:
        conv = layers.BatchNormalization(axis=-1)(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv2D(n_filters, (kernel_size, kernel_size), padding='same')(conv)
    if batchnorm:
        conv = layers.BatchNormalization(axis=-1)(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(n_filters, (1, 1), padding='same')(input_tensor)
    if batchnorm:
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

    result = layers.add([shortcut, conv])
    result = layers.Activation('relu')(result)
    return result


def attention_decoder_block(bottom_tensor, skip_tensor, n_filters: int, kernel_size: int, upsample_size: int,
                            dropout: float, batchnorm: bool, residual: bool = False):
    gating = gating_signal(bottom_tensor, n_filters=n_filters, batchnorm=batchnorm)
    attention = attention_block(skip_tensor, gating, n_filters=n_filters)
    up = layers.UpSampling2D((upsample_size, upsample_size))(bottom_tensor)
    up = layers.concatenate([up, attention], axis=-1)
    if residual:
        up = residual_conv_block(up, n_filters=n_filters, kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    else:
        up = conv_block(up, n_filters=n_filters, kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    return up


def Attention_UNet(input_shape, num_classes: int = 1, num_filters: int = 32, dropout: float = 0.0, batchnorm=True,
                   final_activation='sigmoid'):
    filters = [num_filters * 2 ** i for i in range(5)]  # [32, 64, 128, 256, 512]
    kernel_size = 3
    upsample_size = 2

    # Input layer
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='input')

    # Encoder
    conv1, pool1 = encoder_block(inputs, filters[0], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    conv2, pool2 = encoder_block(pool1, filters[1], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    conv3, pool3 = encoder_block(pool2, filters[2], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    conv4, pool4 = encoder_block(pool3, filters[3], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)

    # Bottom
    conv5 = conv_block(pool4, filters[4], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)

    # Decoder
    up1 = attention_decoder_block(conv5, conv4, filters[3], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=False)
    up2 = attention_decoder_block(up1, conv3, filters[2], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=False)
    up3 = attention_decoder_block(up2, conv2, filters[1], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=False)
    up4 = attention_decoder_block(up3, conv1, filters[0], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=False)

    # Output layer
    outputs = layers.Conv2D(num_classes, kernel_size=(1, 1))(up4)
    outputs = layers.BatchNormalization(axis=-1)(outputs)
    outputs = layers.Activation(final_activation)(outputs)

    # Model integration
    model = keras.Model(inputs=inputs, outputs=outputs, name='Attention_U-Net')
    return model


def Attention_ResUNet(input_shape, num_classes: int = 1, num_filters: int = 32, dropout: float = 0.0,
                      batchnorm: bool = True, final_activation='sigmoid'):
    filters = [num_filters * 2 ** i for i in range(5)]  # [32, 64, 128, 256, 512]
    kernel_size = 3
    upsample_size = 2

    # Input layer
    inputs = layers.Input(input_shape, dtype=tf.float32, name='input')

    # Encoder
    conv1 = residual_conv_block(inputs, filters[0], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = residual_conv_block(pool1, filters[1], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = residual_conv_block(pool2, filters[2], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    conv4 = residual_conv_block(pool3, filters[3], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    # Bottom
    conv5 = residual_conv_block(pool4, filters[4], kernel_size=kernel_size, dropout=dropout, batchnorm=batchnorm)

    # Decoder
    up1 = attention_decoder_block(conv5, conv4, filters[3], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=True)
    up2 = attention_decoder_block(up1, conv3, filters[2], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=True)
    up3 = attention_decoder_block(up2, conv2, filters[1], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=True)
    up4 = attention_decoder_block(up3, conv1, filters[0], kernel_size=kernel_size, upsample_size=upsample_size,
                                  dropout=dropout, batchnorm=batchnorm, residual=True)

    # Output layer
    outputs = layers.Conv2D(num_classes, kernel_size=(1, 1))(up4)
    outputs = layers.BatchNormalization(axis=-1)(outputs)
    outputs = layers.Activation(final_activation)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Attention_ResU-Net")
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


def SuimNet(input_shape, num_classes, **kwargs):
    def concat_skip(layer_input, skip_input, filters, f_size=3):
        u = layers.Conv2D(filters, (f_size, f_size), strides=1, padding='same', activation='relu')(layer_input)
        u = layers.BatchNormalization(momentum=0.8)(u)
        u = layers.Concatenate()([u, skip_input])
        return u

    # Input
    inputs = layers.Input(shape=input_shape, name='input')

    # Encoder
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

    # Decoder
    # First decoder block
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

    # Output
    output = layers.Conv2D(num_classes, (3, 3), strides=1, padding='same', activation='sigmoid')(decoded3_skip)

    model = keras.Model(inputs=inputs, outputs=output, name='SuimNet')
    return model
