import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture('video.avi')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

IMAGE_SIZE = (1, 1)
PATH_TO_CKPT = 'embuh.pb'
TEST_IMAGE_PATHS = 'test'
PATH_TO_LABELS = os.path.join('data', 'label.pbtxt')
offset_height= 1
offset_width = 1
NUM_CLASSES = 1
target_height = 1
target_width = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      im = Image.fromarray(np.uint8(image_np))
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
              
      for i, box in enumerate(np.squeeze(boxes)): #iterasi keberapa??
          if(np.squeeze(scores)[i] > 0.5):          
            ymin = boxes[0,i,0]
            xmin = boxes[0,i,1]
            ymax = boxes[0,i,2]
            xmax = boxes[0,i,3]
            (im_width, im_height) = im.size
            (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
            #cv2.rectangle(image_np,(int(xminn),int(yminn)),(int(xmaxx), int(ymaxx)),(0,255,0),3) #point 
            #Point explain
            #(pt1,pt2)---------------------------|
            #|                                   |
            #|                                   |
            #|                                   |
            #|                                   |
            #|                                   |
            #|________________________________(pt3,pt4)
            rm1pt1 = 0
            rm1pt2 = 200
            rm1pt3 = 100
            rm1pt4 = 300

            rm2pt1 = 110
            rm2pt2 = 200
            rm2pt3 = 210
            rm2pt4 = 300

            rm3pt1 = 290
            rm3pt2 = 200
            rm3pt3 = 390
            rm3pt4 = 300

            rm4pt1 = 400
            rm4pt2 = 200
            rm4pt3 = 500
            rm4pt4 = 300

            rm5pt1 = 510
            rm5pt2 = 200
            rm5pt3 = 610
            rm5pt4 = 300


              
            #cv2.rectangle(image_np,(int(rm1pt1),int(rm1pt2)),(int(rm1pt3), int(rm1pt4)),(0,0,0),3)
            #cv2.rectangle(image_np,(int(rm2pt1),int(rm2pt2)),(int(rm2pt3), int(rm2pt4)),(0,0,0),3)
            #cv2.rectangle(image_np,(int(rm3pt1),int(rm3pt2)),(int(rm3pt3), int(rm3pt4)),(0,0,0),3)
            #cv2.rectangle(image_np,(int(rm4pt1),int(rm4pt2)),(int(rm4pt3), int(rm4pt4)),(0,0,0),3)
            #cv2.rectangle(image_np,(int(rm5pt1),int(rm5pt2)),(int(rm5pt3), int(rm5pt4)),(0,0,0),3)

            centerX = (xminn+xmaxx)/2
            centerY = (yminn+ymaxx)/2
            
            cv2.circle(image_np,(int(centerX),int(centerY)), 5, (0,0,255), -1)
            '''
            if int(rm1pt1) < centerX < int(rm1pt1)+int(rm1pt3) and int(rm1pt2) < centerY < int(rm1pt2)+int(rm1pt4):
              #cv2.rectangle(image_np,(int(rm1pt1),int(rm1pt2)),(int(rm1pt3), int(rm1pt4)),(0,0,0),3)
            else:
              #cv2.rectangle(image_np,(int(rm1pt1),int(rm1pt2)),(int(rm1pt3), int(rm1pt4)),(255,0,0),3)


            if int(rm2pt1) < centerX < int(rm2pt1)+int(rm2pt3) and int(rm2pt2) < centerY < int(rm2pt2)+int(rm2pt4):
              #cv2.rectangle(image_np,(int(rm2pt1),int(rm2pt2)),(int(rm2pt3), int(rm2pt4)),(0,0,0),3)
            else:
              #cv2.rectangle(image_np,(int(rm2pt1),int(rm2pt2)),(int(rm2pt3), int(rm2pt4)),(255,0,0),3)


            if int(rm3pt1) < centerX < int(rm3pt1)+int(rm3pt3) and int(rm3pt2) < centerY < int(rm3pt2)+int(rm3pt4):
              #cv2.rectangle(image_np,(int(rm3pt1),int(rm3pt2)),(int(rm3pt3), int(rm3pt4)),(0,0,0),3)
            else:
              #cv2.rectangle(image_np,(int(rm3pt1),int(rm3pt2)),(int(rm3pt3), int(rm3pt4)),(255,0,0),3)

            if int(rm2pt1) < centerX < int(rm4pt1)+int(rm4pt3) and int(rm4pt2) < centerY < int(rm4pt2)+int(rm4pt4):
              #cv2.rectangle(image_np,(int(rm4pt1),int(rm4pt2)),(int(rm4pt3), int(rm4pt4)),(0,0,0),3)
            else:
              #cv2.rectangle(image_np,(int(rm4pt1),int(rm4pt2)),(int(rm4pt3), int(rm4pt4)),(255,0,0),3)

            if int(rm5pt1) < centerX < int(rm5pt1)+int(rm5pt3) and int(rm5pt2) < centerY < int(rm5pt2)+int(rm5pt4):
              #cv2.rectangle(image_np,(int(rm5pt1),int(rm5pt2)),(int(rm5pt3), int(rm5pt4)),(0,0,0),3)
            else:
              #cv2.rectangle(image_np,(int(rm5pt1),int(rm5pt2)),(int(rm5pt3), int(rm5pt4)),(255,0,0),3)
            #cv2.rectangle(image_np,(int(xminn),int(yminn)),(int(xmaxx), int(ymaxx)),(100,100,0),3)
            #cv2.rectangle(image_np,(int(xminn),int(yminn)),(int(xmaxx), int(ymaxx)),(0,100,100),3)
            #if
            '''
            img_data = sess.run(cropped_image)
            img_data0 = img_data[:,:,::-1].copy()
            cv2.imshow('Crop',img_data0)
            
      #cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      cv2.imshow('object detection', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break