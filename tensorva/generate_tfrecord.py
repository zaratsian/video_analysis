
####################################################################################################
#
#   Convert Pascal VOC (XMLs) to TFRecord format 
#
#   Used with Tensorflow Object Detection API. This code is required whenever a new tensorflow
#   label is being trained (either from scrath or through transfer learning)
#
#   Usage: 
#       ./generate_tfrecord.py <dir_images> <dir_annotations> <output_path> <label_map.json>
#
#
# From the tensorflow/models/research/ directory
'''
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/home/dzaratsian/tensorva/data/ssd_mobilenet_v2_coco.config \
    --train_dir=/home/dzaratsian/tensorva/train
'''
#
#
# Directory Structure
'''
.
./data
./data/train.record
./data/tensorflow_va_training.pbtxt
./data/eval.record
./models
./models/model.ckpt.meta
./models/model.ckpt.data-00000-of-00001
./models/model.ckpt.index
./train
./train/graph.pbtxt
./train/pipeline.config
./train/events.out.tfevents.1529617540.b14afc5d139f
./train/model.ckpt-0.index
./train/checkpoint
./train/model.ckpt-0.data-00000-of-00001
./ssd_mobilenet_v2_coco.config
'''
#
#   NOTE: The .config file must match the type of model that is being used for transfer learning (ie. from the Model Zoo)
#
#       The .config file can be copied from here, based on the model that you'd like to use for transfer learning: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
#       Addiditional info on the config file: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md
#
#       For Transfer learning, use ckpt files, found here:  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#
####################################################################################################


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os,re
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

flags = tf.app.flags
flags.DEFINE_string('dir_images', '/tmp/images', 'Images directory')
flags.DEFINE_string('dir_annotations', '/tmp/images_labeled', 'Pascal VOC Annotations (as directory of XML files)')
flags.DEFINE_string('output_path', '/tmp/train.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS


####################################################################################################
#
#   Functions
#
####################################################################################################

def convert_pascal_voc_xml_to_tfrecord(xml_filepath, dir_images):
    
    # Label Map should link an ID (int) to each category that we are training against.
    # label_map.json
    label_map = {
        'jetta': 1
    }
    
    xmldata    = re.sub('(\n|\r|\t)',' ',open(xml_filepath,'r').read())
    
    filename   = re.sub('<.*?>','',re.findall('<filename>.*?</filename>',xmldata)[0])
    path       = re.sub('\/$','',dir_images) + '/' + str(filename)
    #path      = re.sub('<.*?>','',re.findall('<path>.*?</path>',xmldata)[0])
    
    # Read in IMG and convert to encoded IMG
    with tf.gfile.GFile(path , 'rb') as fid:
        encoded_jpg = fid.read()
    
    # Below is only used if I want to derive width and height based on image (otherwise I'll pull this from the XML)
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = Image.open(encoded_jpg_io)
    #width, height = image.size
    
    img_format = filename.split('.')[-1].encode('utf8')
    height     = int(re.sub('<.*?>','',re.findall('<height>.*?</height>',xmldata)[0]))
    width      = int(re.sub('<.*?>','',re.findall('<width>.*?</width>',xmldata)[0]))
    
    labels     = [re.sub('<.*?>','',i).encode('utf8') for i in re.findall('<name>.*?</name>',xmldata)]
    labels_int = [label_map[label.decode('utf-8')] for label in labels]
    xmins      = [float(re.sub('<.*?>','',i))/float(width)  for i in re.findall('<xmin>.*?</xmin>',xmldata)]
    xmaxs      = [float(re.sub('<.*?>','',i))/float(width)  for i in re.findall('<xmax>.*?</xmax>',xmldata)]
    ymins      = [float(re.sub('<.*?>','',i))/float(height) for i in re.findall('<ymin>.*?</ymin>',xmldata)]
    ymaxs      = [float(re.sub('<.*?>','',i))/float(height) for i in re.findall('<ymax>.*?</ymax>',xmldata)]
    # Used for testing (this is the format that is expected)
    #labels     = [b'jetta']
    #labels_int = [1]
    #xmins      = [0.3]
    #xmaxs      = [0.4]
    #ymins      = [0.2]
    #ymaxs      = [0.3]
    
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

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    xml_files = [ re.sub('\/$','',FLAGS.dir_annotations) + '/' + str(xmlfile) for xmlfile in os.listdir(FLAGS.dir_annotations)]
    
    for xml_filepath in xml_files:
        tf_example = convert_pascal_voc_xml_to_tfrecord(xml_filepath, FLAGS.dir_images)
        writer.write(tf_example.SerializeToString())
    
    writer.close()
    print('Successfully created the TFRecords: {}'.format(FLAGS.output_path))


if __name__ == '__main__':
    tf.app.run()



#ZEND
