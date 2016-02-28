#!/usr/bin/env python
import os
import argparse
from mytensorflow.data.tfrecord import convert_to

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

convert_to(args.root, args.output_file, size=args.size)
