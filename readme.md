# Pex Machine Learning Technical Challenge #

Part 1 - Create a labeled image dataset with two classes:
1. Indoor photographs (e.g. Bedrooms, Bathrooms, Classrooms, Offices)
2. Outdoor photographs (e.g. Landscapes, Skyscrapers, Mountains, Beaches)
Download a subset of examples from the YouTube-8M labeled video dataset:
https://research.google.com/youtube8m/explore.html
Extract relevant frames from the videos to build a balanced dataset of indoor and outdoor
images. The dataset should contain a few thousand images in total. This task can be performed
with tools like OpenCV or FFmpeg.
Create a train/test split of the data.

Part 2 - Train an image classifier​ capable of detecting if a scene is ​indoors or outdoors​:
The model can be an artificial neural network or another approach of your choosing.
Evaluate the accuracy using the ratio of ​ all true results / the total number of examples tested​ .

Part 3 - Create an evaluation tool for classifying single images with the trained model:
This CLI tool needs to allow for image files to be input one at a time.
The output needs to return a string containing the predicted class label.


## Requirements ##
Code is written for Python 3.7. 

Python package requirements you will find in __requirements.txt__, 
```pip instal -r requirements.txt```

Tested on __Ubuntu 19.10__ and __Debian 10___; for __Windows / MacOS__ would be necessary some changes in bash 
scripts for data preparation (video.download_youtube_url_segment:109)

If you will prepare data you also would need __ffmpeg__ installed on your system. 

The package __tensorflow__ is necessary 
only for extracting data, not for training and eval phase (training and eval of model
 is written in __Chainer__ toolkit).

# 1. Data #

Data download and preparation. You can skip this and download prepared dataset (zipped images)
[directly from here](https://drive.google.com/file/d/1Pk7MEDnUqW1oX8atJnihpn1hOPHh5nw4/view?usp=sharing). Possibly, you will need to fix the paths in data/*.csv files.

This data were prepared as follows:

## 1.1 Get data from Youtube 8M dataset ##

We will download the validation set with annotated segments (as described on 
[Youtube 8m dataset webpage](https://research.google.com/youtube8m/download.html))

```curl data.yt8m.org/download.py | partition=3/frame/validate mirror=eu python```

and 1/10 of training data (should be enough for our challenge.) This is annotated only on video level.

```curl data.yt8m.org/download.py | shard=1,10 partition=2/video/train mirror=eu python```

## 1.2 Filter out videos only with indoor and outdoor ##

Now, we filter out the videos only with desired indoor/outdoor videos.
I choose labels as follows (with the most counts in dataset and the most distinguished in terms 
indoor/outdoor):

OUTDOORS = "Highway,Forest,Lake,Desert,Mountain,Building,House,Tree,River,Beach,Garden,City

INDOORS = "Room,Bar,Restaurant,Home improvement,Kitchen,Living room,Gym,Classroom,Office"
 
In output list we have also the URLs of the videos and (if known) time segments.
 
```
python prepare_data_extract.py /path/to/downloaded/tfrecords data/vocabulary.csv data/data_to_download.csv
#python prepare_data_extract.py <dir with downloaded dataset *.tfrecord> <downloaded vocabulary from dataes dataset> <output file>
```

If we check downloaded videos/labels, we see that is more or less balanced in terms of 
 indoor/outdoor (55%/45%).
 
 ![Histogtam of topics](https://github.com/dansoutner/pex-ml-challenge/blob/master/data/dataset_topic_hist.png)
 
 By closer look I decided to get rid of _House_ images from dataset, because 
 this label included a lot of videos from inside and outside of the house,
  which is confusing for model. I excluded also videos which also have label _Game_ 
  (dataset includes also some videos from playing Minecraft etc.). And I also decided to lower count of _gym_ images (compare to 
 room label = a lot of videos of people in gym :) ). Afterwards, we have balanced dataset with 50% 
 of indoor pictures and 50% of outdoor pictures.  
 
## 1.2 Download appropriate chunks of videos ##

Let's download the data. We don't want to download the whole videos, so we download only 2 seconds segments.
If we know segment, we use the segment times, otherwise we take two segments from 1/3 and 2/3 of video 
(videos are usually couple minutes long, so we get more data).
We are using csv list from previous step.

```
python prepare_data_download.py data/data_to_download.csv data/video/ data/videos.csv
```

## 1.3 Convert to images and split ##

Next step is to convert videos to images, we take simply the first
 frame of video.

```
python prepare_data_convert.py data/videos.csv data/img/ data/img.csv
```

We split them to train/validation/test folds (90%/5%/5%):

```
python shuffle_and_split_data.py data/img.csv 0.1 data/img_dev+test.csv data/img_train.csv
python shuffle_and_split_data.py data/img_dev+test.csv 0.5 data/img_dev.csv data/img_test.csv
```

We compute mean image from our train data (this usually helps a model to better fit):

```python compute_mean_image.py data/train.csv -o data/mean_train_image.npz```

We subtract this mean image from every image in data and then we scale data to the values between 0. and 1. (we do this on-the-fly, code in _dataset.py_)

# 2. Train model #

It requires the training and validation dataset of following format:
- Each line contains one training example.
- Each line consists of two elements separated by space(s).
- The first element is a path to RGB image.
- The second element is its ground truth label (integer, in our case 0 or 1).
This is output from data preparation steps (section 1) or you download
 this [from here.](https://drive.google.com/file/d/1Pk7MEDnUqW1oX8atJnihpn1hOPHh5nw4/view?usp=sharing)

We pre-process __train__ data on-the-fly as follows: 

1. All images are scaled so that shorter size is 224px.
2. We randomly crop image to square 224x224.
3. With probability 50% we flip image horizontally.
4. Scale values to interval (0, 1)

We pre-process __dev__ and __test__ on-the-fly data as follows: 

1. All images are scaled so that shorter size is 224px.
2. And then we randomly crop images to square 224x224.
3. Scale values to interval (0, 1)

We use neural network inspired by VGG net, but significantly smaller
 (our dataset is smaller and time 
for training limited). The model configuration could be changed in _VGGnet.py_

```python train.py data/train.csv data/dev.csv --mean-image mean.npy --batchsize 32 --gpu-id 0 --max-epoch 20```

With my very basic setup, it trains in 9 epoch to about 84% accuracy on validation set.

![Loss during training](https://github.com/dansoutner/pex-ml-challenge/blob/master/models/loss.png)

![Accuracy during training](https://github.com/dansoutner/pex-ml-challenge/blob/master/models/accuracy.png)

# 3. Evaluate model #

## 3.1 Evaluate model on all test data ##

For evaluating on test dataset we run eval.py script.
The provided trained model on my simple test (about 600 images) has 
performance __about 80%__ (measured in accuracy).   

It requires the test dataset of following format (csv file):
- Each line contains one training example.
- Each line consists of two elements separated by space(s).
- The first element is a path to RGB image.

```python eval.py data/test.csv model/vggsmall_1.npz ```

## 3.2 Pass one image to the model ##
For evaluating one input image run 

```python eval_one.py data/samples/img.tiff models/model9 --mean-image models/mean.npy```

where parameters are 
```
<input image file> <model file> --mean/image <mean computed image file>
```

Some sample images are stored in _data/samples_.
