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


def download_video_chunks(input_csv_file, output_data_path, output_list):
	"""
	Download data chunks specified in csv file VIDEO_ID, YT_URL, SEGMENT_START, SEGMENT_END
	:param input_csv_file:
	:param output_data_path:
	:param output_list: output list with downloaded files and correspond labels
	"""

	os.makedirs(output_data_path, exist_ok=True)

	# data with segments
	df = pd.read_csv(input_csv_file, keep_default_na=False, sep=",")

	# open output
	with open(output_list, "w") as fout:

		fout.write("file_name label\n")

		for index, row in df.iterrows():
			url = row['yt_url']
			vid_id = row["video_id"]
			seg_start = row["segment_start"]
			seg_end = row["segment_end"]
			if row["in/out"] == "out":
				label = 0
			else:
				label = 1

			# filter out bad urls
			if len(url) == 0:
				continue

			# skip already created videos
			if (os.path.exists(os.path.join(output_data_path, vid_id+"_"+str(index)+"A.mp4")) or
				os.path.exists(os.path.join(output_data_path, vid_id+"_"+str(index)+"B.mp4")) or
				os.path.exists(os.path.join(output_data_path, vid_id+"_"+str(index)+".mp4"))):
				continue

			if seg_start == "" or seg_end == "":
				# if we have no segment information, so take two segments from 1/3 and 2/3 of video
				vid_duration = video.get_yt_video_lenght(url)
				vid_fnameA = os.path.join(output_data_path, vid_id+"_"+str(index)+"A")
				vid_fnameB = os.path.join(output_data_path, vid_id+"_"+str(index)+"B")
				if not video.download_youtube_url_segment(url, vid_fnameA, vid_duration/3, 2):
					logging.warning("Error during downloading file %s occurred." % vid_fnameA)
				else:
					fout.write("{} {}\n".format(vid_fnameA, label))
				if not video.download_youtube_url_segment(url, vid_fnameB, vid_duration*(2/3), 2):
					logging.warning("Error during downloading file %s occurred." % vid_fnameB)
				else:
					fout.write("{} {}\n".format(vid_fnameB, label))
			else:
				# we download whole specified segment
				vid_fname = os.path.join(output_data_path, vid_id+"_"+str(index))
				if not video.download_youtube_url_segment(url, vid_fname, int(seg_start), int(seg_end)-int(seg_start)):
					logging.warning("Error during downloading file %s occurred." % vid_fname)
				else:
					fout.write("{} {}\n".format(vid_fname, label))


if __name__ == "__main__":

	DESC = """
	Download videos specified in csv file
	VIDEO_ID, YT_URL, SEGMENT_START, SEGMENT_END (in seconds), LABEL
	e.g. Az8i,https://www.youtube.com/watch?v=bb5q6cToH8Q,406,150,155
	If you do not provide segment_or segment_end times, we take short segments from 1/3 and 2/3 of video
	
	Output list (on every line) 
	VIDEO_FILE_PATH, LABEL 
	"""

	parser = argparse.ArgumentParser(description=DESC)
	parser.add_argument('input_csv_file', help='input csv file path')
	parser.add_argument('output_dir', help='output path')
	parser.add_argument('output_list', help='output csv file')
	args = parser.parse_args()

	# download video chunks data
	download_video_chunks(args.input_csv_file, args.output_dir, args.output_list)

