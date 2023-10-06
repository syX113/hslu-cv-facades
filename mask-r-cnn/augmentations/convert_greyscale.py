import detectron2.data.transforms as T
import cv2
import numpy as np

# Custom Augmentation to convert images to greyscale
class CustomAugmentationGreyscale(T.Augmentation):
    def get_transform(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Expand the channel dimension to make it a 3 channel grayscale image.
        gray_image = np.stack([gray_image, gray_image, gray_image], axis=-1)
        return T.ColorTransform(lambda x: gray_image)