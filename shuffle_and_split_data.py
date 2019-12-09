#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import numpy as np
import logging
import argparse
import pandas as pd


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def main(input_list, output_list1, output_list2, ratio, shuffle=True):

	# load data
	df = pd.read_csv(input_list, header=0, sep=" ")

	# shuffle
	if shuffle:
		df = df.reindex(np.random.permutation(df.index))

	# get two folds
	count_rows = len(df.index)
	split = int(count_rows * ratio)
	df1 = df.iloc[0:split]
	df2 = df.iloc[split:]

	# write out
	df1.to_csv(output_list1, sep=' ', index=False)
	df2.to_csv(output_list2, sep=' ', index=False)


if __name__ == '__main__':

	DESC = """
	Shuffle data and split them into two folds
	"""

	parser = argparse.ArgumentParser(description=DESC)
	parser.add_argument('input_list', help='input list of files')
	parser.add_argument('ratio', default=0.5, type=float, help='Ratio of data folds')
	parser.add_argument('output_list1', help='output list')
	parser.add_argument('output_list2', help='output list')
	parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle data on/off')
	args = parser.parse_args()

	main(args.input_list, args.output_list1, args.output_list2, args.ratio, shuffle=args.shuffle)