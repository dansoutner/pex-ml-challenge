#!/usr/bin/env python
import argparse
import sys

import numpy as np

import dataset


def compute_mean(dataset):
	"""
	Compute mean image from input dataset
	"""
	sum_image = 0
	N = len(dataset)
	N_real = 0
	for i, (image, _) in enumerate(dataset):
		if i == 0:
			sum_image = image
			N_real += 1
		elif sum_image.shape == image.shape:
			sum_image += image
			N_real += 1
		sys.stderr.write('{} / {}\r'.format(i, N))
		sys.stderr.flush()
	sys.stderr.write('\n')
	return sum_image / N_real


def main():
	parser = argparse.ArgumentParser(description='Compute images mean array')
	parser.add_argument('dataset',
						help='Path to training image-label list file')
	parser.add_argument('--output', '-o', default='mean.npy',
						help='path to output mean array')
	args = parser.parse_args()

	data = dataset.PreprocessOnTheFlyDataset(args.dataset, np.zeros((3, 224, 224)), 224, random_flip=False, random_crop=True)
	mean = compute_mean(data) * 255
	np.save(args.output, mean)


if __name__ == '__main__':
	main()
