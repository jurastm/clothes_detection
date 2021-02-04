import numpy as np
import tensorflow as tf
import cv2
import os
import pkg_resources

from .yolov3_tf2.models import YoloV3

WEIGHTS_PATH = pkg_resources.resource_filename('clothes_detection', 'graphs/deepfashion2_yolov3')
STORE_IMG_PATH = pkg_resources.resource_filename('clothes_detection', 'img1.jpg')

CLASS_NAMES = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
              'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
              'long_sleeve_dress', 'vest_dress', 'sling_dress']

def Read_Img_2_Tensor(img_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3, dtype=tf.dtypes.float32)
    img = tf.expand_dims(img, 0) # fake a batch axis

    return img

class ClothesDetector(object):

    def __init__(self):

        self.model = YoloV3(classes=13)
        self.model.load_weights(WEIGHTS_PATH)

    def __call__(self, image):
        
        if type(image) == np.ndarray:
            cv2.imwrite(STORE_IMG_PATH, image)
            image_tensor = Read_Img_2_Tensor(STORE_IMG_PATH)
        
        elif type(image) == str and os.path.isfile(image):
            image_tensor = Read_Img_2_Tensor(image)

        else:
            raise TypeError("either should be file path to image or numpy array.")
            
        #if image.shape != (416, 416, 3):
        #    image = cv2.resize(image, (416, 416))

        #image_tensor = tf.expand_dims(image, 0) # fake a batch axis
        #image_tensor = tf.cast(image_tensor, tf.float32) / 255.
        image_tensor = tf.image.resize(image_tensor, (416, 416))
        boxes, scores, classes, nums = self.model(image_tensor)
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()

        if os.path.exists(STORE_IMG_PATH):
            os.remove(STORE_IMG_PATH)

        return boxes, scores, classes, nums




