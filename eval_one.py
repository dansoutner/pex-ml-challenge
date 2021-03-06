#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer

import argparse
import numpy as np
import os
import cv2
import logging

from VGGnet import VGGNetsmall2
import dataset


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def eval(image_fname, model_file, mean_image_file="mean.npy", img_size=224):
	"""
	Loads image and decide with trained model if it is indoors / outdoors.
	We test image 10 times with different random crops to achieve better results.

	:param image_fname: path to image
	:param model_file: path to model file
	:param mean_image_file: path to mean image, if used by model training
	:param img_size: size of image for model input
	:return: probabilities of outdoor / indoor
	"""
	# load mean image if we have this
	if os.path.exists(mean_image_file):
		mean = np.load(mean_image_file)
		logging.info("Loading mean image file %s" % mean_image_file)
	else:
		mean = np.ones((3, img_size, img_size)) * 128

	# load model
	model = VGGNetsmall2()
	chainer.serializers.load_npz(model_file, model)

	# load image
	image = np.asarray(cv2.imread(image_fname), dtype=np.float32)

	# we take image 10-times randomly, cropped
	with chainer.using_config('train', False):
		image = dataset.scale(image, smaller_size=img_size)
		result = []
		for _ in range(1):
			img_in = dataset.center_square_crop(image, img_size)
			#img_in = dataset.random_square_crop(image, img_size)
			img_in = img_in.transpose(2, 0, 1)
			img_in -= mean
			img_in *= (1.0 / 255.0)  # Scale to [0, 1]

			y = model(np.asarray([img_in], dtype=np.float32))
			y = chainer.functions.softmax(y)
			result.append(y.data)

		# and return the mean value from this 10 results
		return np.mean(result, axis=0)


def main():
	parser = argparse.ArgumentParser(description='Evaluate model on one image and decide if it is indoor / outdoor')
	parser.add_argument('image_file',
						help='Path to training image-label list file')
	parser.add_argument('model_file',
						help='Path to model file')
	parser.add_argument('--mean-image', help='File with precomputed mean image')
	args = parser.parse_args()

	res = eval(args.image_file, args.model_file, mean_image_file=args.mean_image,)

	if res[0][0] > res[0][1]:
		print("Outdoor (%.2f)" % res[0][0])
	else:
		print("Indoor: (%.2f)" % res[0][1])

if __name__ == "__main__":
	main()
