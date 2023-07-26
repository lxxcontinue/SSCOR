# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import time
from PIL import Image
import argparse
Image.MAX_IMAGE_PIXELS = 2300000000


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""sample"""
    )
    parser.add_argument("--in_dir", type=str, default='./raw_image/fig4', help="the image path")
    parser.add_argument("--out_dir", type=str, default='sample_result', help="the save path of samping patches")
    parser.add_argument("--img_name", type=str, default='SRS_stripe.tif', help="image name")

    parser.add_argument("--patch_size", type=int, default=256, help='the patch size')

    parser.add_argument("--x_loc", type=int, default=0, help='the x location of SFA')

    return parser.parse_args()


args = get_arguments()

img_name = args.img_name
vx = args.patch_size
vy = args.patch_size
(filename, extension) = os.path.splitext(img_name)
in_dir = args.in_dir
out_dir = os.path.join(args.out_dir, f"sample_{filename}")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

out_dir_A = os.path.join(out_dir, 'trainA')
out_dir_B = os.path.join(out_dir, 'trainB')
if not os.path.exists(out_dir_A):
    os.makedirs(out_dir_A)
    os.makedirs(out_dir_B)
    os.makedirs(os.path.join(out_dir, 'testA'))
    os.makedirs(os.path.join(out_dir, 'testB'))

input_file = os.path.join(in_dir, img_name)
im = Image.open(input_file)
img_arr = np.array(im)

dx = vx//2
dy = vy//2

arr1 = img_arr.copy()
arr2 = img_arr.copy()

arr1_B = img_arr.copy()
arr2_B = img_arr.copy()


def cut():

    n = 1
    x1 = 0
    y1 = 0
    x2 = x1 + vx
    y2 = y1 + vy

    xx = im.size[1] - vx

    x_loc = args.x_loc

    while x2 <= x_loc:
        while y2 <= im.size[0]:
            img_A = im.crop((y1, x1, y2, x2))
            img_B = im.crop((y1, xx, y2, xx + vx))

            arr1[x1:x2, y1:y2, :] = arr1[x1:x2, y1:y2, :] // 2

            imgA_name = filename + "_" + str(y1) + '_' + str(x1) + '.tif'
            out_file_A = os.path.join(out_dir_A, imgA_name)
            img_A.save(out_file_A)

            imgB_name = filename + "_" + str(y1) + '_' + str(x1) + '_B' + '.tif'
            out_file_B = os.path.join(out_dir_B, imgB_name)
            img_B.save(out_file_B)

            if n % 50 == 0:
                print("processing")

            y1 = y1 + dy
            y2 = y1 + vy
            n = n + 1

        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy
        xx = xx - int((dx / x_loc) * (im.size[1] - x_loc))

    img = Image.fromarray(arr1)
    out_file = os.path.join(out_dir, "crop.tif")
    img.save(out_file)


if __name__ == "__main__":
    cut()
