import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np


class RCNN:

    def __init__(self):
        self.ssObject = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def get_iou(self, bb1, bb2):
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
        iou = intersection_area / \
            float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def crop_image(self, img, tol=0):
        # img is 2D image data
        # tol  is tolerance
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def transform_voc_box_to_pixel_coordinate(self, box, width, height):
        ymin, xmin, ymax, xmax = box
        xmin *= width
        xmax *= width
        ymin *= height
        ymax *= height
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def crop_image_and_adapt_boxes(self, inputImage, input_train_boxes, indexImage):
        image = self.crop_image(cv2.cvtColor(
            inputImage, cv2.COLOR_RGB2GRAY))
        origImg = inputImage
        imout = origImg[0:image.shape[0], 0:image.shape[1], :]
        bboxes = input_train_boxes[indexImage]
        height, width = imout.shape[:2]
        converted_boxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = self.transform_voc_box_to_pixel_coordinate(
                bbox, width, height)
            converted_boxes.append([xmin, ymin, xmax, ymax])
        converted_boxes = np.array(converted_boxes)
        return imout, converted_boxes

    def rcnn_generator(self, input_train_images, input_train_boxes, input_labels, aug, IMG_SIZE, BATCH_SIZE):
        # generator have to loop forever
        while True:
            train_images = []
            train_labels = []
            while len(train_images) < BATCH_SIZE:
                # extra image and remove padding
                for indexImage, inputImage in enumerate(input_train_images):
                    if (len(train_images) > BATCH_SIZE):
                        break
                    imout, converted_boxes = self.crop_image_and_adapt_boxes(
                        inputImage, input_train_boxes, indexImage)
                    # Selective Search
                    self.ssObject.setBaseImage(imout)
                    self.ssObject.switchToSelectiveSearchFast()
                    ssresults = self.ssObject.process()
                    flag = 0
                    counter_foreground = 0
                    counter_background = 0
                    # Loop over the first 2000 result of Selective Search
                    for e, result in enumerate(ssresults):
                        if (len(train_images) > BATCH_SIZE):
                            break
                        x, y, w, h = result
                        if e < 2000 and flag == 0:
                            for indexBox, gtval in enumerate(converted_boxes):
                                if gtval[0] == gtval[1] == gtval[2] == gtval[3] == 0:
                                    continue
                                iou = self.get_iou(
                                    {"x1": gtval[0], "x2": gtval[2], "y1": gtval[1], "y2": gtval[3]}, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                                if counter_foreground < BATCH_SIZE/2:
                                    if iou > 0.70 and indexBox < len(input_labels[indexImage]) and input_labels[indexImage][indexBox] != 0:
                                        resized_and_cropped = cv2.resize(
                                            imout[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                                        train_images.append(
                                            resized_and_cropped)
                                        train_labels.append(
                                            input_labels[indexImage][indexBox]+1)
                                        counter_foreground += 1
                                if counter_background < BATCH_SIZE/2:
                                    if iou < 0.3:
                                        resized_and_cropped = cv2.resize(
                                            imout[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                                        train_images.append(
                                            resized_and_cropped)
                                        train_labels.append(0)
                                        counter_background += 1

                            if counter_foreground >= BATCH_SIZE/2 and counter_background >= BATCH_SIZE/2:
                                flag = 1
                                break
            if aug is not None:
                (images, labels) = next(aug.flow(np.array(train_images),
                                                 np.array(train_labels), batch_size=BATCH_SIZE))
            yield (np.array(images), np.array(labels))

    def build_generator(self, IMG_SIZE, BATCH_SIZE, trainList, testList):
        x_train, y_train, boxes_train = trainList
        x_test, y_test, boxes_test = testList
        aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                 horizontal_flip=True, fill_mode="nearest")

        # initialize both the training and testing image generators
        # I use only the first 500 images for space reasons. In case you have multiple GPUs or a
        # GPU with more memory than mine feel free to change this value
        train_gen = self.rcnn_generator(x_train[0:500, :, :, :],
                                        boxes_train[0:500, :, :], y_train[0:500, :], aug, IMG_SIZE, BATCH_SIZE)
        val_gen = self.rcnn_generator(x_test[0:500, :, :, :],
                                      boxes_test[0:500, :, :], y_test[0:500, :], aug, IMG_SIZE, BATCH_SIZE)
        return train_gen, val_gen

    @staticmethod
    def build(self, IMG_SIZE, BATCH_SIZE, num_classes=20):

        vggmodel = VGG16(weights='imagenet', include_top=True)

        for layers in (vggmodel.layers)[:15]:
            layers.trainable = False
        X = vggmodel.layers[-2].output
        predictions = Dense(num_classes, activation="softmax")(X)
        model = Model(inputs=vggmodel.input, outputs=predictions)
        return model
