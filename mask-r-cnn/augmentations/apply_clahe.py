import detectron2.data.transforms as T
import cv2
import numpy as np

# Custom Augmentation to apply CLAHE to the images
class CustomAugmentationCLAHE(T.Augmentation):
    def apply_clahe(self, image):
        channels = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
        enhanced_channels = [clahe.apply(channel) for channel in channels]
        enhanced_image = cv2.merge(enhanced_channels)
        return enhanced_image

    def get_transform(self, image):
        enhanced_image = self.apply_clahe(image)
        return T.ColorTransform(lambda x: enhanced_image)