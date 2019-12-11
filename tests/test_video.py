import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import video
import logging


logging.disable(logging.CRITICAL)


class TestVideo(unittest.TestCase):

	def setUp(self):
		# remove all test files if any
		for ext in (".mp4", ".avi", ".foo", ".mkv", ".tiff", ".jpg"):
			fname = "test" + ext
			if os.path.exists(fname):
				os.remove(fname)

	def test_get_image_from_first_video_frame(self):
		ret = video.get_image_from_first_video_frame("video_samples/PIky_1107A.mp4", "test.tiff")
		self.assertTrue(ret)
		ret = video.get_image_from_first_video_frame("video_samples/PIky_1107A.mp4", "test.jpg")
		self.assertTrue(ret)
		

	def test_get_video_length(self):
		ret = video.get_video_length("video_samples/PIky_1107A.mp4")
		self.assertAlmostEqual(ret, 2.3, places=1)
		ret = video.get_video_length("video_samples/non/existing/video")
		self.assertIsNone(ret)


	def test_get_yt_video_length_value(self):
		self.assertEqual(video.get_yt_video_length("https://www.youtube.com/watch?v=NqQ1a1n4wnk"), 182)
		self.assertIsNone(video.get_yt_video_length("https://www.foo.com"))
		self.assertRaises(TypeError, video.get_yt_video_length, None, 0)

	def test_get_yt_video_length_inputs(self):
		self.assertIsNone(video.get_yt_video_length("XXX"))
		self.assertIsNone(video.get_yt_video_length(0))
		self.assertIsNone(video.get_yt_video_length(True))
		self.assertIsNone(video.get_yt_video_length(None))

	def test_get_youtube_url_value(self):
		self.assertEqual(video.get_youtube_url("SXCZ"), "https://www.youtube.com/watch?v=NqQ1a1n4wnk")

	def test_get_youtube_url_inputs(self):
		self.assertIsNone(video.get_youtube_url("XXX"))
		self.assertIsNone(video.get_youtube_url(0))
		self.assertIsNone(video.get_youtube_url(True))
		self.assertIsNone(video.get_youtube_url(None))

	def test_download_youtube_url_segment_mp4(self):

		ret = video.download_youtube_url_segment("https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test.mp4", 0, 1)
		self.assertTrue(ret)
		self.assertTrue(os.path.exists("test.mp4"))

		ret = video.download_youtube_url_segment("https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test", 0, 1)
		self.assertTrue(ret)
		self.assertTrue(os.path.exists("test.mp4"))

	def test_download_youtube_url_segment_avi(self):

		ret = video.download_youtube_url_segment("https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test", 0, 1, video_type=".avi")
		self.assertTrue(ret)
		self.assertTrue(os.path.exists("test.avi"))

	def test_download_youtube_url_segment_unknown_type(self):

		ret = video.download_youtube_url_segment("https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test", 0, 1, video_type=".foo")
		self.assertFalse(ret)
		self.assertFalse(os.path.exists("test.foo"))

	def test_download_youtube_url_segment_url(self):

		ret = video.download_youtube_url_segment("https://www.foo.com", "test", 0, 1, video_type=".mp4")
		self.assertFalse(ret)

	def test_download_youtube_url_segment_inputs(self):

		ret = video.download_youtube_url_segment(None, "test.mp4", 0, 1)
		self.assertFalse(ret)

		ret = video.download_youtube_url_segment(0, "test.mp4", 0, 1)
		self.assertFalse(ret)

		self.assertRaises(AttributeError, video.download_youtube_url_segment, "https://www.youtube.com/watch?v=NqQ1a1n4wnk", None, 0, 1)
		self.assertRaises(AttributeError, video.download_youtube_url_segment, "https://www.youtube.com/watch?v=NqQ1a1n4wnk", 0, 0, 1)
		self.assertRaises(TypeError, video.download_youtube_url_segment, "https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test.mp4", "0", "1")

	def test_download_youtube_url_segment_segment_values(self):

		ret = video.download_youtube_url_segment("https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test.mp4", 0.0, 1.1)
		self.assertTrue(ret)

		# should not be that log, thi video, but ffmpeg returns True
		ret = video.download_youtube_url_segment("https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test.mp4", 100000000, 10000001)
		self.assertTrue(ret)

		ret = video.download_youtube_url_segment("https://www.youtube.com/watch?v=NqQ1a1n4wnk", "test.mp4", -100000000, 10000001)
		self.assertFalse(ret)


if __name__ == '__main__':
	unittest.main()
