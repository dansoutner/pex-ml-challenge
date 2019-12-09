#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Loading general libraries
import os
import pandas as pd
import logging
import argparse

# Local imports
import video


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def convert_all_videos_in_dir_to_images(input_list_fname, output_dir, output_list_fname, img_type=".tiff"):
	"""
	Convert all videos in INPUT_LIST to images and flush to OUTPUT_DIR
	:param input_list_fname: list of files to convert (on every line VIDEO_FILE_PATH, LABEL)
	:param output_dir: flush output files here
	:param output_list_fname: list of output files (on every line VIDEO_FILE_PATH, LABEL)
	"""
	os.makedirs(output_dir, exist_ok=True, )

	df = pd.read_csv(input_list_fname, keep_default_na=False, sep=" ")

	with open(output_list_fname, "w") as fout:
		fout.write("file_name label\n")

		for index, row in df.iterrows():
			fname = row['file_name']
			label = row["label"]
			foutname = os.path.join(output_dir, os.path.basename(fname) + img_type)

			try:
				video.get_image_from_first_video_frame(fname, foutname)
				fout.write("{} {}\n".format(foutname, label))

			except (OSError, IOError):
				logging.warning("Can not convert file %s to image." % fname)


if __name__ == "__main__":

	DESC = """Convert all videos from input list (on every line we have):
	 VIDEO_FILE_PATH, LABEL
	 to images (extracting first frame from video).
	 
	 Ouptut list (on every lien one)
	 IMAGE_FILE_PATH, LABEL
	 """

	parser = argparse.ArgumentParser(description=DESC)
	parser.add_argument('input_list', help='input csv list with indexes and labels')
	parser.add_argument('output_dir', help='output path')
	parser.add_argument('output_list', help='output scp file')
	args = parser.parse_args()

	# convert all videos to images
	convert_all_videos_in_dir_to_images(args.input_dir, args.input_list, args.output_dir, args.output_list)


