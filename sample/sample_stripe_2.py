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
        description="""sample both horizontal and vertical stripes"""
    )
    parser.add_argument("--h_n", type=int, default=1, help='the number of horizontal stripe')
    parser.add_argument("--v_n", type=int, default=1, help='the number of vertical stripe')

    parser.add_argument("--in_dir", type=str, default='./raw_image/fig4', help="the image path")
    parser.add_argument("--out_dir", type=str, default='sample_result', help="the save path of samping patches")
    parser.add_argument("--img_name", type=str, default='SRS_stripe.tif', help="image name")

    parser.add_argument("--patch_size", type=int, default=256, help='the patch size')
    parser.add_argument("--direction", type=int, default=0, help='Direction of stripe change,[0, 1, 2, 3]: [Upper Left, Upper Right, Lower Left, Lower Right]')

    return parser.parse_args()


args = get_arguments()

img_name = args.img_name
h_n = args.h_n
v_n = args.v_n
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

arr1 = img_arr.copy()
arr2 = img_arr.copy()
arr3 = img_arr.copy()

arr1_B = img_arr.copy()
arr2_B = img_arr.copy()
arr3_B = img_arr.copy()

length = int(im.size[0] / v_n)
width = int(im.size[1] / h_n)

dx = vx//2
dy = vy//2

direction_list = [[-1, -1], [1, -1], [-1, 1], [1, 1]]

direction_control = direction_list[args.direction]
B_x_y_h = [0, vy * direction_control[1]]
B_x_y_v = [vx * direction_control[0], 0]
corner_move_pix = [[-50, -50], [0, 0]]

junction_range_list = []
for i in range(max(h_n, v_n)):
    if i == 0:
        continue
    r = range(i * length - vx//2, i * length + vx//2)
    junction_range_list.append(r)


def cut_h():

    n = 1
    x1 = int(width - vx / 2)
    y1 = 0
    x2 = x1 + vx
    y2 = y1 + vy

    y_flag = 1

    while x2 <= im.size[1]:

        while y2 <= im.size[0]:
            img_A = im.crop((y1, x1, y2, x2))

            arr1[x1:x2, y1:y2, :] = arr1[x1:x2, y1:y2, :] // 2

            B_y = B_x_y_h[0] + int(y1)
            B_x = B_x_y_h[1] + int(x1)

            for rr in junction_range_list:
                yy1, yy2 = min(rr), max(rr)

                if yy1 in range(y1, y2):
                    B_y = y1 - vy
                    B_x = x1 + vx * direction_control[1]
                elif yy2 in range(y1, y2):
                    B_y = y1 + vy
                    B_x = x1 + vx * direction_control[1]

            if 0 <= B_y + vy < im.size[0] and 0 <= B_x + vx < im.size[1]:
                img_B = im.crop((B_y, B_x, B_y + vy, B_x + vx))
                arr1_B[B_x:B_x + vx, B_y:B_y + vy, :] = arr1_B[B_x:B_x + vx, B_y:B_y + vy, :] // 2

                imgA_name = filename + "_" + str(y1) + '_' + str(x1) + '.tif'
                out_file_A = os.path.join(out_dir_A, imgA_name)
                img_A.save(out_file_A)

                imgB_name = filename + "_" + str(y1) + '_' + str(x1) + '_B' + '.tif'
                out_file_B = os.path.join(out_dir_B, imgB_name)
                img_B.save(out_file_B)

            if n % 50 == 0:
                print("h processing")

            y1 = y1 + dy
            y2 = y1 + vy
            n = n + 1

            # 最后不足vy尺寸的patch也要
            if y2 > im.size[0] and y_flag:
                y_flag = 0
                y1 = im.size[0] - vy
                y2 = y1 + vy

        x1 = x1 + width
        x2 = x1 + vx
        y1 = 0
        y2 = vy
        y_flag = 1

    img = Image.fromarray(arr1)
    out_file = os.path.join(out_dir, "crop-h.tif")
    img.save(out_file)

    img = Image.fromarray(arr1_B)
    out_file = os.path.join(out_dir, "crop-h-B.tif")
    img.save(out_file)


def cut_v():

    n = 1
    x1 = 0
    y1 = int(length - vy/2)
    x2 = x1 + vx
    y2 = y1 + vy

    x_flag = 1

    while y2 <= im.size[0]:

        while x2 <= im.size[1]:
            img_A = im.crop((y1, x1, y2, x2))

            arr2[x1:x2, y1:y2, :] = arr2[x1:x2, y1:y2, :] // 2

            B_y = B_x_y_v[0] + int(y1)
            B_x = B_x_y_v[1] + int(x1)

            for rr in junction_range_list:
                xx1, xx2 = min(rr), max(rr)

                if xx1 in range(x1, x2):
                    B_y = y1 + vy * direction_control[0]
                    B_x = x1 - vx
                elif xx2 in range(x1, x2):
                    B_y = y1 + vy * direction_control[0]
                    B_x = x1 + vx

            if 0 <= B_y + vy < im.size[0] and 0 <= B_x + vx < im.size[1]:
                img_B = im.crop((B_y, B_x, B_y + vy, B_x + vx))
                arr2_B[B_x:B_x + vx, B_y:B_y + vy, :] = arr2_B[B_x:B_x + vx, B_y:B_y + vy, :] // 2


                imgA_name = filename + "_" + str(y1) + '_' + str(x1) + '.tif'
                out_file_A = os.path.join(out_dir_A, imgA_name)
                img_A.save(out_file_A)

                imgB_name = filename + "_" + str(y1) + '_' + str(x1) + '_B' + '.tif'
                out_file_B = os.path.join(out_dir_B, imgB_name)
                img_B.save(out_file_B)

            if n % 50 == 0:
                print("v processing")

            x1 = x1 + dx
            x2 = x1 + vx
            n = n + 1

            # 最后不足vx尺寸的patch也要
            if x2 > im.size[1] and x_flag:
                x_flag = 0
                x1 = im.size[1] - vx
                x2 = x1 + vx

        y1 = y1 + length
        y2 = y1 + vy
        x1 = 0
        x2 = vx
        x_flag = 1

    img = Image.fromarray(arr2)
    out_file = os.path.join(out_dir, "crop-v.tif")
    img.save(out_file)

    img = Image.fromarray(arr2_B)
    out_file = os.path.join(out_dir, "crop-v-B.tif")
    img.save(out_file)


def cut_corners():

    n = 1
    x1 = 0
    y1 = 0
    x2 = x1 + vx
    y2 = y1 + vy

    while x2 <= im.size[1]:

        while y2 <= im.size[0]:

            for i, move in enumerate(corner_move_pix):
                yy = y1 + move[0]
                xx = x1 + move[1]

                if yy < 0 or xx < 0:
                    continue

                img_A = im.crop((yy, xx, yy+vy, xx+vx))

                img_B = im.crop((yy-2*vy, xx-2*vx, yy-vy, xx-vx))
                if yy-2*vy < 0 or xx-2*vx < 0:
                    continue

                arr3[xx:xx + vx, yy:yy + vy, :] = arr2[xx:xx + vx, yy:yy + vy, :]//2
                arr3_B[xx-2*vx:xx-vx, yy-2*vy:yy-vy, :] = arr3_B[xx-2*vx:xx-vx, yy-2*vy:yy-vy, :]//2


                imgA_name = filename + "_" + str(yy) + '_corner_' + str(xx) + '.tif'
                out_file_A = os.path.join(out_dir_A, imgA_name)
                img_A.save(out_file_A)

                imgB_name = filename + "_" + str(yy) + '_corner_' + str(xx) + '_B' + '.tif'
                out_file_B = os.path.join(out_dir_B, imgB_name)
                img_B.save(out_file_B)

                if n % 50 == 0:
                    print("corner processing")

            y1 = y1 + length
            y2 = y1 + vy
            n = n + 1

        x1 = x1 + width
        x2 = x1 + vx
        y1 = 0
        y2 = y1 + vy

    img = Image.fromarray(arr3)
    out_file = os.path.join(out_dir, "crop-c.tif")
    img.save(out_file)

    img = Image.fromarray(arr3_B)
    out_file = os.path.join(out_dir, "crop-c-B.tif")
    img.save(out_file)


if __name__ == "__main__":

    cut_h()
    cut_v()
    cut_corners()


