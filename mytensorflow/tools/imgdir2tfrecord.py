#!/usr/bin/env python
import os, sys
import argparse
import yaml
import cv2
import tensorflow as tf
from mytensorflow.data.imgdir import ImageDirectory
from mytensorflow.data.tfrecord import encode_image

parser = argparse.ArgumentParser()
parser.add_argument("root", help="path to root of image directory")
parser.add_argument("--output", help="path to write output tfrecord to")
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--train_split", type=float, default=0.8)
parser.add_argument('--size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
args = parser.parse_args()

if args.size is None:
    print "Must specify a size for the images with --size W H"
    sys.exit(1)

args.size = tuple(args.size)

if args.output is None:
    args.output = os.path.basename(args.root)

imgdir = ImageDirectory(args.root)
train_filename = args.output + ".train.tfrecord"
test_filename = args.output + ".test.tfrecord"

train, test = imgdir.train_test_split(train=args.train_split)

def resize_and_write(data_split, size, output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for i, (image_file, label) in enumerate(data_split):
        if i % 100 == 0:
            print "Processed %d / %d" % (i, len(data_split))
        image = cv2.resize(cv2.imread(image_file), size)
        encode_image(writer, image, label)

resize_and_write(train, args.size, train_filename)
resize_and_write(test, args.size, test_filename)

# Write metadata file.
labels_filename = os.path.basename(args.root) + ".labels"
num_classes = imgdir.num_classes()
num_images = imgdir.num_images()

metadata_filename = os.path.basename(args.root) + '.metadata'
metadata = {'shape': [args.size[0], args.size[1], 3],
            'num_classes': imgdir.num_classes(),
            'num_train_examples': len(train),
            'num_test_examples': len(test),
            'train': [train_filename],
            'test': [test_filename]}

with open(metadata_filename, 'w') as fp:
    yaml.dump(metadata, fp)

