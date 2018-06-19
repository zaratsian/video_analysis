
####################################################################################################
#
#   Tensorflow Object Detection API
#
####################################################################################################

import os,sys
import numpy as np
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import six.moves.urllib as urllib
import tarfile
import zipfile

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils # as vis_util

####################################################################################################
#
#   Input Vars
#
####################################################################################################

pb_model         = '/opt/models/research/object_detection/test_ckpt/ssd_inception_v2.pb'
pb_model_labels  = '/opt/models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES      = 100000000  # A large number can be used, unless there's a small known number of categories.

TEST_IMAGE_PATHS = ['/Users/dzaratsian/Desktop/images/frame26.jpg']

####################################################################################################
#
#   Load (frozen) Tensorflow model into memory
#
####################################################################################################

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

####################################################################################################
#
#   Load Label Map
#
####################################################################################################

label_map       = label_map_util.load_labelmap(pb_model_labels)
categories      = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_map    =  dict([(cat['id'],cat['name']) for cat in categories])

####################################################################################################
#
#   Image Object Detection
#
####################################################################################################

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)







def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      
      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  
  return output_dict





for image_path in TEST_IMAGE_PATHS:
    image    = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Score image against Model
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    # Print Results to Screen - Used for Terminal
    print('\n[ INFO ] Results for ' + str(image_path))
    for i, score in enumerate(output_dict['detection_scores'].tolist()):  # Get all objects (scores, classes, and boxes) if score >= XX threshold.
        if score >= 0.70:
            category_label = category_map[output_dict['detection_classes'][i]]
            print( str(round(score,6)) + '\t' + str(category_label) + '\t\t' + str(output_dict['detection_boxes'][i])  )
    
    # Visualization of the results of a detection.
    '''
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    '''





#ZEND
