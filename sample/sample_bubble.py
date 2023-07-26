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

    parser.add_argument('--h', action='store_true', help='if true, exist horizontal stripe')
    parser.add_argument('--v', action='store_true', help='if true, exist vertical stripe')
    parser.add_argument("--h_n", type=int, default=1, help='the number of horizontal stripe')
    parser.add_argument("--v_n", type=int, default=1, help='the number of vertical stripe')

    parser.add_argument("--in_dir", type=str, default='./raw_image/fig4', help="the image path")
    parser.add_argument("--out_dir", type=str, default='sample_result', help="the save path of samping patches")
    parser.add_argument("--img_name", type=str, default='SRS_stripe.tif', help="image name")

    parser.add_argument("--patch_size", type=int, default=256, help='the patch size')

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

dx = vx//2
dy = vy//2

h_move_pix = [[0, -dx+50]]
v_move_pix = [[-dy+50, 0]]

length = int(im.size[0] / v_n)
width = int(im.size[1] / h_n)

arr1 = img_arr.copy()
arr2 = img_arr.copy()

arr1_B = img_arr.copy()
arr2_B = img_arr.copy()

bubble_loc = np.array([[6100, 3200, 6650, 3700], [4010, 700, 4610, 1150], [4200, 1750, 4900, 2450], [3800, 3900, 4600, 4250],
                  [2750, 1400, 3400, 1800], [1850, 2950, 2600, 3550], [2650, 3650, 3450, 4200], [950, 2850, 1600, 3350]])


def cut_h():

    n = 1
    x1 = 0
    y1 = 0
    x2 = x1 + vx
    y2 = y1 + vy

    while x2 <= im.size[1]:
        while y2 <= im.size[0]:
            for i, move in enumerate(h_move_pix):
                yy = y1 + move[0]
                xx = x1 + move[1]
                if yy < 0 or xx < 0:
                    continue
                img_A = im.crop((yy, xx, yy+vy, xx+vx))

                B_y = yy
                B_x = xx - vx//2-50

                for _, x_y in enumerate(bubble_loc):
                    b_y1, b_x1, b_y2, b_x2 = x_y[:]
                    if (b_x1 < B_x < b_x2 and b_y1 < B_y < b_y2) or (b_x1 < B_x + vx < b_x2 and b_y1 < B_y + vy < b_y2):
                        B_x = B_x + width
                        if (b_x1 < B_x < b_x2 and b_y1 < B_y < b_y2) or (b_x1 < B_x + vx < b_x2 and b_y1 < B_y + vy < b_y2):
                            B_x = B_x + width
                        break

                img_B = im.crop((B_y, B_x, B_y+vy, B_x+vx))

                arr1[xx:xx+vx, yy:yy+vy, :] = arr1[xx:xx+vx, yy:yy+vy, :] // 2
                arr1_B[B_x:B_x + vx, B_y:B_y + vy, :] = arr1_B[B_x:B_x + vx, B_y:B_y + vy, :] // 2

                imgA_name = filename + "_" + str(yy) + '_' + str(xx) + extension
                out_file_A = os.path.join(out_dir_A, imgA_name)
                img_A.save(out_file_A)

                imgB_name = filename + "_" + str(yy) + '_' + str(xx) + '_B' + extension
                out_file_B = os.path.join(out_dir_B, imgB_name)
                img_B.save(out_file_B)

                if n % 50 == 0:
                    print("processing")

            y1 = y1 + dy
            y2 = y1 + vy
            n = n + 1

        x1 = x1 + width
        x2 = x1 + vx
        y1 = 0
        y2 = vy

    img = Image.fromarray(arr1)
    out_file = os.path.join(out_dir, "crop-h.tif")
    img.save(out_file)

    img = Image.fromarray(arr1_B)
    out_file = os.path.join(out_dir, "crop-h-B.tif")
    img.save(out_file)


if __name__ == "__main__":
    cut_h()
