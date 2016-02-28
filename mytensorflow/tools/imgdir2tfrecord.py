#!/usr/bin/env python
import os
import argparse
import yaml
from mytensorflow.data.tfrecord import imgdir_to_tfrecord

parser = argparse.ArgumentParser()
parser.add_argument("root", help="path to root of image directory")
parser.add_argument("--output_file", help="path to write output tfrecord to")
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument('--size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
args = parser.parse_args()

if args.size is not None:
    args.size = tuple(args.size)

if args.output_file is None:
    args.output_file = os.path.basename(args.root) + ".tfrecord"

num_images = imgdir_to_tfrecord(args.root, args.output_file, size=args.size)

# Write metadata file.
labels_filename = os.path.basename(args.root) + ".labels"
with open(labels_filename) as fp:
    num_classes = len(fp.readlines())

metadata_filename = os.path.basename(args.root) + '.metadata'
metadata = {'shape': [args.size[0], args.size[1], 3],
            'num_classes': num_classes,
            'num_examples': num_images,
            'filenames': [args.output_file]}

with open(metadata_filename, 'w') as fp:
    yaml.dump(metadata, fp)
