
########################################################################################################
#
#   Converts a image directory and label (XML) directory into a tensorflow object for Model Training.
#
#   Usage: python convert_to_tfrecord.py
#
########################################################################################################

import os,re
import io
import xml.etree.ElementTree as ET
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image


def create_tf_example(image_filename):
    
    image_path  = '/tmp/images/'         + str(image_filename)
    labels_path = '/tmp/images_labeled/' + re.sub('\.[a-zA-Z]{3,5}$','',image_filename) + '.xml'
    
    # Read the image
    img = Image.open(image_path)
    width, height = img.size
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=img.format)
    
    height = height
    width = width
    encoded_image_data = img_bytes.getvalue()
    image_format = img.format.encode('utf-8')
    
    # Read the label XML
    tree = ET.parse(labels_path)
    root = tree.getroot()
    xmins = xmaxs = ymins = ymaxs = list()
    
    for coordinate in root.find('object').iter('bndbox'):
        xmins = [int(coordinate.find('xmin').text)]
        xmaxs = [int(coordinate.find('xmax').text)]
        ymins = [int(coordinate.find('ymin').text)]
        ymaxs = [int(coordinate.find('ymax').text)]
    
    classes_text = ['jetta'.encode('utf-8')]
    classes = [1]
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(encoded_image_data),
        'image/source_id': dataset_util.bytes_feature(encoded_image_data),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


if __name__ == '__main__':
    writer = tf.python_io.TFRecordWriter('train.record')
    for image_filename in os.listdir('/tmp/images/'):
        tf_example = create_tf_example(image_filename)
        writer.write(tf_example.SerializeToString())
    
    writer.close()


#ZEND
