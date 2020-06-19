"""
Sections of this code were taken from the official object detection tutorial:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
"""
import numpy as np

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util

from utils import visualization_utils as vis_util

import cv2

# Path to frozen detection graph.
# for the object detection.
PATH_TO_CKPT = '/home/nijeri/TensorFlow/workspace/training_demo/trained-inference-graphs/output_inference_graph_300x300_sc_mobilenet.pb/frozen_inference_graph.pb'

# path to label map
PATH_TO_LABELS = os.path.join('/home/nijeri/TensorFlow/workspace/training_demo/annotations', 'label_map.pbtxt')

NUM_CLASSES = 1

sys.path.append("..")


def detect_in_video():

    # VideoWriter is  responsible for the creation of a copy of the video
    # including detections overlays.
    # output frame size has to be the same as the input video.
    out = cv2.VideoWriter('/home/nijeri/TensorFlow/workspace/training_demo/yellow_pad_full_2_mobile_520x520_60fps.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 60, (520, 520))

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # definition of input and output Tensors of the detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents an object that was detected
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent level of confidence of the detected objects (bees in this case).
            # Score is shown on the result image, together with the class label
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            cap = cv2.VideoCapture('/home/nijeri/TensorFlow/workspace/training_demo/yellow_pad_full_2.MP4')

            while(cap.isOpened()):
                # Read the frame
                ret, frame = cap.read()

                # Recolor the frame. By default, OpenCV uses BGR color space.
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #crop frame with focus on landing pad y1:y2 , x1:x2

                #settings for format 700x700 
                color_frame = color_frame[200:720, 400:920]

                #settings for format 300x300 
                #color_frame = color_frame[300:620, 500:820]

                image_np_expanded = np.expand_dims(color_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    min_score_thresh=.30)

                cv2.imshow('frame', color_frame)
                output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                out.write(output_rgb)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out.release()
            cap.release()
            cv2.destroyAllWindows()


def main():
    detect_in_video()


if __name__ == '__main__':
    main()