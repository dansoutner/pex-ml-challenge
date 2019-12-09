#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Loading general libraries
import tensorflow as tf
import os
import pandas as pd
import logging
import argparse

# Local imports
import video

# Setting variables
OUTDOORS = set("Highway Forest Lake Desert Mountain Building House Tree River Beach Garden City".split())  # Street?
INDOORS = set("Room,Bar,Restaurant,Home improvement,Kitchen,Living room,Gym,Classroom,Office".split(","))


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def load_subset_vocab(fname_vocabulary):
	"""
	Load
	:param: file with dataset vocabulary (provided with dataest)
	:return: list of indoor and outdoor indexes
	"""
	vocab = pd.read_csv(fname_vocabulary)
	logging.info("we have {} unique labels in the dataset".format(len(vocab['Index'].unique())))

	label_mapping = vocab[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']
	label_mapping_w = vocab[['Name', 'Index']].set_index('Name', drop=True).to_dict()['Index']

	# subset for in/outdoors
	outdoors_i = set([label_mapping_w[i] for i in OUTDOORS])
	indoors_i = set([label_mapping_w[i] for i in INDOORS])
	return outdoors_i, indoors_i


def extract_segments_from_yt8m_dataset(tfrecord_dir, vocabulary_file, out_fname):
	"""
	extract videos and segments that we are interested in from whole YT8M dataset
	:param: dir which includes tfrecords with data from YT8M
	:param: path to vacabulary file
	:return: path to csv file with segments
	"""

	i = 0

	outdoors_i, indoors_i = load_subset_vocab(vocabulary_file)

	# open files to log data
	with open(out_fname, "w") as fout:

		# write header of output .csv file
		# id of video in dataset YT8M, youtube URL, label, our indoor/outdoor label, start segment time if we know, end segment time
		fout.write("video_id,yt_url,label,in/out,segment_start,segment_end\n")

		# walk all files in dataset dir
		for file_count, fname in enumerate(os.listdir(tfrecord_dir)):

			# all videos in file
			for example in tf.python_io.tf_record_iterator(os.path.join(tfrecord_dir, fname)):

				# get data
				tf_example = tf.train.Example.FromString(example)
				tf_seq_example = tf.train.SequenceExample.FromString(example)
				vid_id = tf_seq_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
				val_vid_labels = tf_seq_example.context.feature['labels'].int64_list.value
				segment_start_times = tf_seq_example.context.feature['segment_start_times'].int64_list.value
				segment_end_times = tf_seq_example.context.feature['segment_end_times'].int64_list.value
				segment_labels = tf_seq_example.context.feature['segment_labels'].int64_list.value
				segment_scores = tf_seq_example.context.feature['segment_scores'].float_list.value
				n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

				# write segments to our lists
				if len(set(segment_labels) & outdoors_i) > 0:
					for seg_idx, seg in enumerate(segment_labels):
						if (seg in outdoors_i) and (segment_scores[seg_idx] > 0.9):
							fout.write("%s,%s,%d,'out',%d,%d\n"
							               % (vid_id, video.get_youtube_url(vid_id), seg, segment_start_times[seg_idx],
							                  segment_end_times[seg_idx]))
					i += 1

				# indoor part
				if len(set(segment_labels) & indoors_i) > 0:
					for seg_idx, seg in enumerate(segment_labels):
						if (seg in indoors_i) and (segment_scores[seg_idx] > 0.9):
							fout.write("%s,%s,%d,'in',%d,%d\n"
							              % (vid_id, video.get_youtube_url(vid_id), seg, segment_start_times[seg_idx],
							                 segment_end_times[seg_idx]))
					i += 1


if __name__ == "__main__":

	DESC = """
	Iterates over all .tfrecord files from dataset and extracts videos and video-segments
	which are labelled as outdoor / indoor.
	Returns .csv file with columns as
	video_id,yt_url,label,in/out,segment_start,segment_end
	(id of video in dataset YT8M), (youtube URL), (label), (our indoor/outdoor label), (start segment time if we know), (end segment time)
	"""

	parser = argparse.ArgumentParser(description=DESC)
	parser.add_argument('input_tfrecord_dir', help='input path')
	parser.add_argument('vocabulary', help='dataset vocabulary path')
	parser.add_argument('output_list', help='output list path')
	args = parser.parse_args()

	# get videos and segments from dataset
	extract_segments_from_yt8m_dataset(args.input_tfrecord, args.vocabulary, args.output_list)

