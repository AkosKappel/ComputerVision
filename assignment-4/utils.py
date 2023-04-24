import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split

# Segmentation classes
class_names = [
    'Background (waterbody)', 'Human divers', 'Aquatic plants and sea-grass',
    'Wrecks and ruins', 'Robots (AUVs/ROVs/instruments)', 'Reefs and invertebrates',
    'Fish and vertebrates', 'Sea-floor and rocks',
]
num_classes = len(class_names)
class_codes = {
    0: [0, 0, 0],  # Black
    1: [0, 0, 1],  # Blue
    2: [0, 1, 0],  # Green
    3: [0, 1, 1],  # Sky
    4: [1, 0, 0],  # Red
    5: [1, 0, 1],  # Pink
    6: [1, 1, 0],  # Yellow
    7: [1, 1, 1],  # White
}
class_weights = np.array([
    1.0591986,
    14.61948805,
    15.71746382,
    5.09713808,
    44.17992558,
    1.0,
    4.78133668,
    2.42209091,
])

# Augmentation
data_augmentation = A.Compose([
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


def load_dataset(train_path, test_path, val_split=0.2, seed=411):
    # Get paths
    train_images_path = os.path.join(train_path, 'images')
    train_masks_path = os.path.join(train_path, 'masks')
    test_images_path = os.path.join(test_path, 'images')
    test_masks_path = os.path.join(test_path, 'masks')

    # Get image and mask paths
    train_val_images = list_files(train_images_path)
    train_val_masks = list_files(train_masks_path)
    test_images = list_files(test_images_path)
    test_masks = list_files(test_masks_path)

    # Split into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=val_split, random_state=seed
    )

    return train_images, val_images, test_images, train_masks, val_masks, test_masks


def list_files(path):
    content = os.listdir(path)
    # Get only files (not directories)
    files = [os.path.join(path, f) for f in content if os.path.isfile(os.path.join(path, f))]
    # Ensure the lists are sorted to match the corresponding image and mask
    files.sort()
    return files


def rgb_to_label(image, label_codes, one_hot=True):
    label_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # Assigns each pixel a label from [0, num_classes - 1]
    for class_num, color in label_codes.items():
        label_image[np.all(image == color, axis=-1)] = class_num
    # Converts labeled image from (H, W) to (H, W, num_classes),
    # where each pixel is a one-hot vector (e.g. [2] => [0, 0, 1, 0, 0, 0, 0, 0])
    if one_hot:
        n_classes = len(label_codes)
        label_image = to_categorical(label_image, n_classes)
    return label_image


def label_to_rgb(image, label_codes, one_hot=True):
    rgb_image = np.empty((image.shape[0], image.shape[1], 3), dtype=np.float32)
    # Assigns each pixel a color from label_codes
    if one_hot:
        for class_num, color in label_codes.items():
            rgb_image[image[:, :, class_num] == 1] = color
    else:
        for class_num, color in label_codes.items():
            rgb_image[image == class_num] = color
    return rgb_image


def load_and_preprocess(image_path, mask_path, width=None, height=None, augmentation=None,
                        normalize=True, keep_rgb_mask=False, one_hoe_encode=False):
    # Load images
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    mask = cv.imread(mask_path, cv.IMREAD_COLOR)

    # Convert to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

    # Resize
    if width is not None and height is not None:
        img = cv.resize(img, (width, height), interpolation=cv.INTER_NEAREST)
        mask = cv.resize(mask, (width, height), interpolation=cv.INTER_NEAREST)

    # Augment
    if augmentation is not None:
        augmented = augmentation(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

    # Fix mask values to be 0 or 255
    if not np.all(np.logical_or(mask == 0, mask == 255)):
        mask = np.round(mask / 255.0) * 255.0

    # Normalize
    if normalize:
        img = (img / 255.0).astype(np.float32)
        mask = (mask / 255.0).astype(np.float32)

    # Return RGB image and mask
    if keep_rgb_mask:
        return img, mask

    # Convert mask to:
    # - label encoded image (each pixel is an integer from 0 to num_classes - 1)
    # - one-hot encoded image (each pixel is a one-hot vector with 1 in the class index)
    mask = rgb_to_label(mask, class_codes, one_hot=one_hoe_encode)
    return img, mask


def iou_score(y_true, y_pred, smooth=1):
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


class DataGenerator(Sequence):

    def __init__(self, image_paths, mask_paths, batch_size, img_height, img_width,
                 augmentation=None, shuffle=True, rgb_mask=True, preprocess_fn=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.rgb_mask = rgb_mask
        self.preprocess_fn = preprocess_fn
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        image_paths_temp = [self.image_paths[k] for k in indexes]
        mask_paths_temp = [self.mask_paths[k] for k in indexes]

        images, masks = self.__data_generation(image_paths_temp, mask_paths_temp)

        return images, masks

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths_temp, mask_paths_temp):
        # Initialize batch arrays
        images = []
        masks = []

        for img_path, mask_path in zip(image_paths_temp, mask_paths_temp):
            # Load and preprocess image and mask
            img, mask = load_and_preprocess(
                img_path, mask_path, width=self.img_width, height=self.img_height,
                augmentation=self.augmentation, keep_rgb_mask=self.rgb_mask,
                normalize=True, one_hoe_encode=True,
            )
            # Add to batch
            images.append(img)
            masks.append(mask)

        # Apply extra preprocessing on input images
        if self.preprocess_fn is not None:
            images = self.preprocess_fn(images)

        return np.array(images), np.array(masks)


class VisualizeProgress(tf.keras.callbacks.Callback):
    def __init__(self, test_data, every_n_epochs=10, num_samples=4, threshold=None, ds=False):
        super(VisualizeProgress, self).__init__()
        self.test_data = test_data
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.threshold = threshold
        self.deep_supervision = ds

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            x_test, y_test = self.test_data
            predictions = self.model.predict(x_test[:self.num_samples])

            if self.deep_supervision:
                predictions = predictions[-1]

            plt.figure(figsize=(12, 12))
            for index in range(self.num_samples):
                if index >= len(x_test): break

                plt.subplot(self.num_samples, 3, index * 3 + 1)
                plt.imshow(x_test[index])
                if index == 0:
                    plt.title('Input Image')

                plt.subplot(self.num_samples, 3, index * 3 + 2)
                if self.threshold is None:
                    ground_truth = label_to_rgb(y_test[index], class_codes, one_hot=True)
                else:
                    ground_truth = y_test[index]
                plt.imshow(ground_truth)
                if index == 0:
                    plt.title('Ground Truth')

                plt.subplot(self.num_samples, 3, index * 3 + 3)
                if self.threshold is None:
                    pred_argmax = np.argmax(predictions[index], axis=-1)
                    prediction = label_to_rgb(pred_argmax, class_codes, one_hot=False)
                else:
                    prediction = (predictions[index] > self.threshold).astype(np.uint8) * 255
                plt.imshow(prediction)
                if index == 0:
                    plt.title('Predicted Segmentation')

            plt.show()
