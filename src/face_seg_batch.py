"""usage: after installing caffe and cloning the repository, do not 
compile cpp source code (unless you want to retrain the model), put
this file in <face_segmentation_dir>/interface/python, and put face_a 
dataset in <face_sementation_dir>/data
"""

import numpy as np
import caffe  # if importing caffe at the end, there is seg-fault on my computer
from PIL import Image
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath
import csv

# only train img are segmented here. Feel free to modify the code.
data_dir = join(dirname(dirname(dirname(abspath(__file__)))), 'data')
img_dir = join(data_dir, 'face_a')

# get image names
fnames = []
with open (join(img_dir , 'train.csv'), 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for fname, _ in csvreader:
        fnames.append(fname)

# caffe init and load net
caffe.set_mode_cpu()
net = caffe.Net(join(data_dir, 'face_seg_fcn8s_deploy.prototxt'), 
                join(data_dir, 'face_seg_fcn8s.caffemodel'), caffe.TEST)

# magic number, I guess this is the mean of training set
MEAN = np.array([104.00698793, 116.66876762, 122.67891434])

BG_COLOR = np.array([0, 0, 0])
# get seg-faces
fig = plt.figure()
for fname in fnames:
    im = Image.open(join(img_dir, 'train', fname))
    im = im.resize((500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= MEAN
    in_ = in_.transpose((2, 0, 1))

    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0) # shape is (500, 500)

    shade_idx = out == 0

    im_ = np.array(im)
    im_[shade_idx] = BG_COLOR

    #fig.add_subplot(1, 2, 1)
    #plt.imshow(im)
    #fig.add_subplot(1, 2, 2)
    plt.imshow(im_)

    # save plots
    out_dir = join(img_dir, 'segmented')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fig.savefig(join(out_dir, 'seg-' + fname))
    plt.clf()






