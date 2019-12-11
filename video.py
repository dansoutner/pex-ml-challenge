#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import subprocess
import urllib
from ast import literal_eval
import pafy
import youtube_dl
#from pytube import YouTube
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def get_image_from_first_video_frame(v_fname, out_fname):
	"""
	Get image from first video frame
	"""
	vidcap = cv2.VideoCapture(v_fname)
	hasFrames, image = vidcap.read()
	if hasFrames:
		cv2.imwrite(out_fname, image)
	else:
		logging.warning("Could not extract image from %s" % v_fname)
	return hasFrames


def get_video_length(fname):
	"""
	Get lenght of video in seconds
	"""
	if not os.path.exists(fname):
		return None
	cap = cv2.VideoCapture(fname)
	fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if fps == 0 or frame_count == 0:
		return None
	duration = frame_count/fps
	cap.release()
	return duration


def get_yt_video_length(url):
	"""
	Get length of YT video without download
	:param url: full youtube video url
	:return: video length in seconds
	"""
	try:
		return pafy.new(url).length
	except (youtube_dl.utils.DownloadError, OSError):
		return 0
	except ValueError:
		return None



def get_youtube_url(vid_id):
	"""
	Get youtube URL from dataset video ID
	:param: video id from dataset
	:return: full url on youtube.com
	"""

	# As described on the YouTube8M download page, for privacy reasons,
	# the video id has been randomly generated and does not directly
	# correspond to the actual YouTube video id. To convert the id into the actual YouTube video id,
	# we follow link: http://data.yt8m.org/2/j/i/UL/UL00.js

	if type(vid_id) is not str:
		return None

	try:
		url = "http://data.yt8m.org/2/j/i/%s/%s.js" % (vid_id[0:2], vid_id)
		contents = urllib.request.urlopen(url).read().decode("utf-8")
		contents = contents[1:-1]
		yt_url = "https://www.youtube.com/watch?v=%s" % literal_eval(contents)[1]
	except urllib.error.HTTPError:
		logging.error("Could not convert video ID to youtube URL")
		yt_url = None
	return yt_url


# def download_youtube_url(yt_url, download_path, video_name=None):
# 	"""
# 	Download the video from YT url
# 	"""
#
# 	yt = YouTube(yt_url)
#
# 	# get the best resolution
# 	resolutions = sorted([stream.resolution for stream in yt.streams.filter(progressive=True).all()])
# 	best_res = resolutions[-1]
#
# 	# filters out all the files with "mp4" extension
# 	video = yt.streams.filter(file_extension='mp4', res=best_res)
#
# 	# if name is None, we make name automatically from URL
# 	if video_name is None:
# 		video_name = yt_url.split("?")[-1]
#
# 	# finally, get the video
# 	video.download(output_path=download_path, filename=video_name)


def download_youtube_url_segment(yt_url, filename, seg_start, seg_duration, video_type=".mp4"):
	"""
	Download only a segment from YT video
	:param yt_url: Youtube url to download
	:param filename: Path where to save output
	:param seg_start: Start time o segemnt in seconds
	:param seg_duration: Duration of video segment in seconds
	:param video_type: Video type, defined by extension for ffmpeg (mp4 mkv avi)
	"""

	# add file.ext for ffmpeg
	if not filename.endswith(video_type):
		filename += video_type

	if video_type not in (".mp4", ".mkv", ".avi"):
		logging.error("Only mp4 mkv avi videos are allowed. (%s, %s)" % (yt_url, video_type))
		return False

	if seg_duration <= 0:
		logging.error("Segment duration should be > 0. (%s, %d)" % (yt_url, seg_duration))
		return False

	if seg_start < 0:
		logging.error("Segment start must be >= 0. (%s, %d)" % (yt_url, seg_start))
		return False

	# all magic happens here:
	# we start downloading form particular time with youtube-dl and continue just for n seconds (ffmpeg)
	cmd = "ffmpeg $(youtube-dl -g '%s' | sed \"s/.*/-ss %d -i &/\") -y -strict -2 -t %d -c copy %s" % (
			yt_url,
			seg_start,
			seg_duration,
			filename,
		)

	try:
		return_code = subprocess.call(cmd, shell=True)
		logging.info(cmd)
	except youtube_dl.utils.ExtractorError:
		return_code = -1

	if return_code == 0:
		logging.info("Successfully downloaded %s from %s" % (filename, yt_url))
	else:
		logging.warning("Can not download file %s from %s" % (filename, yt_url))

	return return_code == 0


if __name__ == "__main__":
	#u = get_youtube_url(vid_ids[13])
	# download_youtube_url(u, ".")
	download_youtube_url_segment("https://www.youtube.com/watch?v=1aheRpmurAou", "foo.mp4", 10, 5)
	get_image_from_first_video_frame("foo.mp4", "foo.jpg")

