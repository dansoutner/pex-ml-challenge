import random
import numpy as np
import chainer
from chainer import datasets
import cv2


class LabeledImageDatasetWithMean(datasets.LabeledImageDataset):
	"""
	Image dataset, we load images from files and subtract mean
	"""

	def __init__(self, path, mean):
		"""
		:param path: path to file with lines IMAGE_FILE_PATH LABEL
		:param mean: array with mean image
		"""
		self.base = datasets.LabeledImageDataset(path)
		self.mean = mean.astype(chainer.get_dtype())

	def __len__(self):
		return len(self.base)

	def get_example(self, i):
		image, label = self.base[i]
		_, h, w = image.shape
		image -= self.mean
		image *= (1.0 / 255.0)  # Scale to [0, 1]
		return image, label


def scale(image, smaller_size=None, inter=cv2.INTER_AREA):
	"""
	Scale image that shorter side will be SMALLER_SIZE
	:param image: image np.array
	:param smaller_size: size of the smaller side of image
	:param inter: cv2 interpolation algorithm
	:return: img np.array
	"""
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if smaller_size is None:
		return image

	# check to see if the width is None
	r = smaller_size / float(min(h, w))
	if h > w:
		dim = (smaller_size, int(h * r))
	elif h == w:
		dim = (smaller_size, smaller_size)
	else:
		dim = (int(w * r), smaller_size)

	# resize the image
	resized = cv2.resize(image, dim, interpolation=inter)

	# return the resized image
	return resized


def imcrop(img, bbox):
	"""
	Safely crop image to bbox
	:param img: np.array with image
	:param bbox: x1, y1, x2, y2 coordinates
	:return: image array
	"""
	x1, y1, x2, y2 = bbox
	if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
		img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
	return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
	"""
	Pad image to fit in bbox
	:param img: np.array with image
	:param x1:
	:param x2:
	:param y1:
	:param y2:
	:return: image np.array
	"""
	img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
	                   (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
	y1 += np.abs(np.minimum(0, y1))
	y2 += np.abs(np.minimum(0, y1))
	x1 += np.abs(np.minimum(0, x1))
	x2 += np.abs(np.minimum(0, x1))
	return img, x1, x2, y1, y2


def random_square_crop(img, size):
	"""
	Crop image to random square with size SIZE
	:param img: input image np.array
	:param size: size of cropped area
	:return: cropped image np.array
	"""
	(h, w) = img.shape[:2]
	if h > w:
		c = random.randint(0, (h - w))
		bbox = (0, c, w, c + size)
	else:
		c = random.randint(0, (w - h))
		bbox = (c, 0, c + size, h)
	return imcrop(img, bbox)


def center_square_crop(img, size):
	"""
	Crop image to center square with size SIZE
	:param img: input image np.array
	:param size: size of cropped area
	:return: cropped image np.array
	"""
	(h, w) = img.shape[:2]
	if h > w:
		c = int((h - size)/2)
		bbox = (0, c, w, c + size)
	else:
		c = int((w - size)/2)
		bbox = (c, 0, c + size, h)
	return imcrop(img, bbox)


class PreprocessOnTheFlyDataset(datasets.LabeledImageDataset):
	"""
	Image dataset with pre-processing
		# It applies following preprocesses:
		#     - Scale to crop size
		#     - Cropping (random or center rectangular)
		#     - Random flip
		#     - Scaling to [0, 1] values
	"""

	def __init__(self, path, mean, crop_size, random_flip=True, random_crop=True):
		"""
		:param path: path to file with lines IMAGE_FILE_PATH LABEL
		:param mean: array with mean image
		:param crop_size: size of output square image
		:param random_flip: flip horizontally
		:param random_crop: crop randomly to (crop_size, crop_size) image
		"""
		self.base = datasets.LabeledImageDataset(path, )
		self.mean = mean.astype(chainer.get_dtype())
		self.crop_size = crop_size
		self.random_flip = random_flip
		self.random_crop = random_crop

	def __len__(self):
		return len(self.base)

	def get_example(self, i):
		# It reads the i-th image/label pair and return a preprocessed image.
		# It applies following preprocesses:
		#     - Scale to crop size
		#     - Cropping (random or center rectangular)
		#     - Random flip
		#     - Scaling to [0, 1] values
		crop_size = self.crop_size

		image, label = self.base[i]
		image = image.transpose(1, 2, 0)
		image = scale(image, smaller_size=crop_size)
		if self.random_crop:
			image = random_square_crop(image, crop_size)
		else:
			image = center_square_crop(image, crop_size)

		image = image.transpose(2, 0, 1)
		# randomly flip, p=0.5
		if self.random_flip and random.randint(0, 1):
			image = image[:, :, ::-1]

		image -= self.mean
		image *= (1.0 / 255.0)  # Scale to [0, 1]
		return image, label