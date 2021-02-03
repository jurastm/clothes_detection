import numpy as np
import tensorflow as tf
import cv2
import pkg_resources

from yolov3_tf2.models import YoloV3

WEIGHTS_PATH = pkg_resources.resource_filename('clothes_detection', 'graphs/deepfashion2_yolov3')

CLASS_NAMES = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
              'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
              'long_sleeve_dress', 'vest_dress', 'sling_dress']

class ClothesDetector(object):

    def __init__(self):

        self.model = YoloV3(classes=13)
        self.model.load_weights(WEIGHTS_PATH)

    def __call__(self, image):

        if image.shape != (416, 416, 3):
            image = cv2.resize(image, (416, 416))

        image_tensor = tf.expand_dims(image, 0) # fake a batch axis
        image_tensor = tf.cast(image_tensor, tf.float32) / 255.

        boxes, scores, classes, nums = self.model(image_tensor)
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()

        return boxes, scores, classes, nums




