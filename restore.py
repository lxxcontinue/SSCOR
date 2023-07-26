import math
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util
from PIL import Image
import matplotlib.pyplot as plt

import datetime

import cv2
import time
import numpy as np
from data.base_dataset import BaseDataset, get_transform

Image.MAX_IMAGE_PIXELS = 2300000000


def judge(img):
    img = img.convert("RGB")
    input_arr = np.array(img)
    arr = input_arr.sum(axis=2)
    if arr.max() == 0:
        # print("it is black")
        return True
    return False


def set_threshold(ori, res, thr):

    arr1 = np.array(ori, dtype="float32")
    arr2 = np.array(res, dtype="float32")

    height = np.min([arr1.shape[0], arr2.shape[0]])
    width = np.min([arr1.shape[1], arr2.shape[1]])
    arr1 = arr1[:height, :width]
    arr2 = arr2[:height, :width]

    # dark region change too much
    arr3 = np.where(arr1 < thr, arr1, arr2)

    # light region change too much
    arr3 = np.where(arr1 > arr3, arr1, arr3)

    return Image.fromarray(np.array(arr3, dtype="uint8"))


def max_combine(im1, im2):
    arr1 = np.array(im1)
    arr2 = np.array(im2)
    arr3 = np.where(arr1 > arr2, arr1, arr2)
    return Image.fromarray(np.array(arr3, dtype="uint8"))


def restore(img):
    n = 0
    per = 0.1

    y_flag = 1
    x_flag = 1

    x1 = 0
    y1 = 0
    x2 = x1 + vx
    y2 = y1 + vy

    one_arr = np.ones((opt.patch_size, opt.patch_size, 3), dtype=np.uint32)

    while x2 <= im.size[1]:

        while y2 <= im.size[0]:
            overlap_arr[x1:x1 + vx, y1:y1 + vy, ] = overlap_arr[x1:x1 + vx, y1:y1 + vy, ] + one_arr

            im2 = img.crop((y1, x1, y2, x2))
            input = im2.convert("RGB")

            if judge(im2):
                y1 = y1 + dy
                y2 = y1 + vy
                if y2 > im.size[0] and y_flag:
                    y_flag = 0
                    y1 = im.size[0] - vy
                    y2 = y1 + vy
                continue

            A_img = input
            transform_A = get_transform(opt, grayscale=(opt.output_nc == 1))
            """A {B,C,H,W} [-1,1]"""
            A = transform_A(A_img).unsqueeze(0)

            img_data = {'A': A}
            model.set_input(img_data)  # unpack data from data loader
            model.test()  # run inference

            visuals = model.get_current_visuals()  # get image results
            img_arr = util.tensor2im(visuals["fake_B"])

            restored_arr[x1:x1 + vx, y1:y1 + vy, ] = restored_arr[x1:x1 + vx, y1:y1 + vy, ] + img_arr

            y1 = y1 + dy
            y2 = y1 + vy

            if y2 > im.size[0] and y_flag:
                y_flag = 0
                y1 = im.size[0] - vy
                y2 = y1 + vy

        n = n + 1
        if n * dx >= per * im.size[1]:
            print('process: {:.0%}'.format(per))
            per = per + 0.1

        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = y1 + vy
        y_flag = 1

        if x2 > im.size[1] and x_flag:
            x_flag = 0
            x1 = im.size[1] - vx
            x2 = x1 + vx

    overlap_arr[overlap_arr == 0] = 1
    overlap_img = Image.fromarray(np.uint8(restored_arr / overlap_arr))

    return overlap_img


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    t1 = time.time()
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    image_name = opt.image_name

    input_file = os.path.join(opt.dataroot, image_name)
    im = Image.open(input_file)

    im_arr = np.array(im)

    overlap_arr = np.zeros_like(im_arr, dtype=np.uint32)
    restored_arr = np.zeros_like(im_arr, dtype=np.uint32)

    ori_im = im.copy()

    vx = opt.patch_size
    vy = opt.patch_size
    dx = opt.offset_size
    dy = opt.offset_size

    save_path = opt.dataroot + "/result/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    overlap_arr = np.zeros_like(im_arr, dtype=np.uint32)
    restored_arr = np.zeros_like(im_arr, dtype=np.uint32)
    res_im = restore(ori_im)
    res_thr_im = set_threshold(im, res_im, opt.dark_threshold)

    for i in range(opt.repeat-1):
        dx = dx + 40
        dy = dy + 40
        print(dx, dy)
        overlap_arr = np.zeros_like(im_arr, dtype=np.uint32)
        restored_arr = np.zeros_like(im_arr, dtype=np.uint32)
        res_im = restore(ori_im)
        res_im = set_threshold(im, res_im, opt.dark_threshold)
        res_thr_im = max_combine(res_thr_im, res_im)

    save_name = "restore-%s" % image_name
    res_thr_im.save(os.path.join(save_path, save_name))

    print("all finish")
    endtime = datetime.datetime.now()
    print('image name:%s, restore time:%.2fs' % (image_name, (time.time() - t1)))



