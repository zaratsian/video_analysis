
####################################################################################################
#
#   Convert Pascal VOC (XMLs) to TFRecord format 
#
#   Used with Tensorflow Object Detection API. This code is required whenever a new tensorflow
#   label is being trained (either from scrath or through transfer learning)
#
#   Usage:
#
#
# From the tensorflow/models/research/ directory
'''
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/tmp/zarats/models/model/tensorflow_va_training.config \
    --train_dir=/tmp/zarats/models/model/train/
'''
#
#   The .config file can be copied from here, based on the model that you'd like to use for transfer learning: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
#       Addiditional info on the config file: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md
#   For Transfer learning, use ckpt files, found here:  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#  
####################################################################################################


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


####################################################################################################
#
#   Args / Inputs
#
####################################################################################################


dir_images      = '/tmp/images'             # Images directory
dir_annotations = '/tmp/images_labeled'     # Pascal VOC Annotations (as directory of XML files)
output_path     = '/tmp/train.record'  


# Label Map should link an ID (int) to each category that we are training against.
label_map = {
    'jetta': 1
}


####################################################################################################
#
#   Functions
#
####################################################################################################

def convert_pascal_voc_xml_to_tfrecord(xml_filepath, label_map, dir_images):
    
    xmldata    = re.sub('(\n|\r|\t)',' ',open(xml_filepath,'r').read())
    
    filename   = re.sub('<.*?>','',re.findall('<filename>.*?</filename>',xmldata)[0])
    path       = re.sub('\/$','',dir_images) + '/' + str(filename)
    #path      = re.sub('<.*?>','',re.findall('<path>.*?</path>',xmldata)[0])
    img_format = filename.split('.')[-1]
    height     = int(re.sub('<.*?>','',re.findall('<height>.*?</height>',xmldata)[0]))
    width      = int(re.sub('<.*?>','',re.findall('<width>.*?</width>',xmldata)[0]))
    
    labels     = [re.sub('<.*?>','',i).encode('utf8') for i in re.findall('<name>.*?</name>',xmldata)]
    labels_int = [label_map[label.decode('utf-8')] for label in labels]
    xmins      = [re.sub('<.*?>','',i) for i in re.findall('<xmins>.*?</xmins>',xmldata)]
    xmaxs      = [re.sub('<.*?>','',i) for i in re.findall('<xmaxs>.*?</xmaxs>',xmldata)]
    ymins      = [re.sub('<.*?>','',i) for i in re.findall('<ymins>.*?</ymins>',xmldata)]
    ymaxs      = [re.sub('<.*?>','',i) for i in re.findall('<ymaxs>.*?</ymaxs>',xmldata)]
    
    # Read in IMG and convert to encoded IMG
    with tf.gfile.GFile(path , 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    #width, height = image.size
    
    # Create TF Payload   
    tf_payload = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(img_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(labels),
        'image/object/class/label': dataset_util.int64_list_feature(labels_int),
    }))
    return tf_payload


####################################################################################################
# 
#   Main
#
####################################################################################################

def main():
    writer = tf.python_io.TFRecordWriter(output_path)
    
    xml_files = [ re.sub('\/$','',dir_annotations) + '/' + str(xmlfile) for xmlfile in os.listdir(dir_annotations)]
    
    for xml_filepath in xml_files:
        tf_example = convert_pascal_voc_xml_to_tfrecord(xml_filepath, label_map, dir_images)
        writer.write(tf_example.SerializeToString())
    
    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    #tf.app.run()
    main()


#ZEND
