#!/usr/bin/env python
# -*- coding: utf-8 -*-


from chainer import iterators
from chainer.training import extensions
from chainer import links as L
import chainer

import argparse
import numpy as np
import os

from VGGnet import VGGNetsmall2
from dataset import PreprocessOnTheFlyDataset


def eval(dataset_list_eval, model_file, mean_image_file="mean.npy", batch_size=64, gpu_id=0, img_size=224):

	if os.path.exists(mean_image_file):
		mean = np.load(mean_image_file)
	else:
		mean = np.ones((3, img_size, img_size)) * 128

	dataset_eval = PreprocessOnTheFlyDataset(dataset_list_eval, mean, img_size, random_flip=False, random_crop=True)
	eval_iter = iterators.SerialIterator(dataset_eval, batch_size, False, False)

	model = VGGNetsmall2()
	chainer.serializers.load_npz(model_file, model)
	model = L.Classifier(model)

	if gpu_id >= 0:
		model.to_gpu(gpu_id)

	with chainer.using_config('train', False):
		result = extensions.Evaluator(eval_iter, model).evaluate()

	print(result)

	return result


def main():
	parser = argparse.ArgumentParser(description='Train model on input ')
	parser.add_argument('dataset_list_eval',
						help='Path to training image-label list file')
	parser.add_argument('model_file',
						help='Path to model file')
	parser.add_argument('--mean-image', help='File with precomputed mean image')
	parser.add_argument('--batch-size', default=32, type=int,
						help='Set batch size')
	parser.add_argument('--gpu-id', default=-1, type=int,
						help='Set GPU id for computation, for CPU set -1')
	parser.add_argument('--img-size', default=224, type=int,
						help='Input image size')
	args = parser.parse_args()

	res = eval(args.dataset_list_eval, args.model_file,
	     mean_image_file=args.mean_image, gpu_id=args.gpu_id,
	     batch_size=args.batch_size, img_size=args.img_size)

	print(res)

if __name__ == "__main__":
	main()
