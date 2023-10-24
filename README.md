
# A deep learning-based stripe self-correction method for stitched microscopic images

[SSCOR Paper](https://www.biorxiv.org/content/10.1101/2023.01.11.523393v1)

## Pytorch implementation of SSCOR

## OS Requirements
- Linux: Ubuntu 18.04
- Python 3.7 + Pytorch 1.7.1
- NVIDIA GPU + CUDA 11.0 CuDNN 8

## Installation Guide
Setting up the development environment (Installation time is about ten minutes)
- To install docker:
  - Install [docker](https://docs.docker.com/install/)
  - Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Pull image from Dockerhub and run:

  - `docker pull lxxcontinue/docker:v1.2.7`

  - `docker run -it --gpus=all -v <local_dir_path>/SSCOR:/workspace/SSCOR lxxcontinue/docker:v1.2.7`, replace <local_dir_path> with your local dir path.

- To run demo
  - `cd SSCOR`


## Raw Image Data
The raw image data are placed in `./raw_image/` directory.

Download the raw image google drive [fig2](https://drive.google.com/file/d/1ft7olxzalz8kQBT6cc3BjmZE4LQmtEhi/view?usp=drive_link), 
[fig3](https://drive.google.com/file/d/10ceCUBRYDQvjrqky2BBIFH7dbpHQthgj/view?usp=drive_link),
[fig4](https://drive.google.com/file/d/1wYSfOxSfZwVknHdlK9dM0YJp_vffIuZj/view?usp=drive_link),
[fig5](https://drive.google.com/file/d/1nO9jJWaVPF2UXtt0iG0zm4TaqwoVz6kN/view?usp=drive_link)

## Source Data
To maximize the reproducibility of our study, we provide the raw data underlying all the figures and tables demonstrated in our paper. These raw data are saved for each figure/table in specific Excel file with multiple labeled sheets, respectively. The Excel files are deposited in the `./source_data/`directory.

## Stripes And Artifacts Correction 
Run the following command to correct the stripes and artifacts:

### Correct the label-free MPM image with non-uniform stripes
```bash
python restore.py --dataroot ./raw_image/fig2 --name MPM_stripe_1 --model sscor --image_name MPM_stripe_1.tif --offset_size 100 --repeat 2
```
- Run time: 93s (on a workstation with a NVIDIA RTX 3090 GPU)
- The corrected result will be saved in: `./raw_image/fig2/result`

We have uploaded a trained model and placed it in the [google drive](https://drive.google.com/file/d/1wKDHx9rp84DEDp0EUONCpFpNbM1OdlWn/view?usp=drive_link). You can save it to './checkpoints/' to restore './raw_image/fig2/MPM_stripe_1.tif'

### Correct the synthetic stripes and out-of-focus artifacts on SRS image
```bash
python restore.py --dataroot ./raw_image/fig4 --name SRS_out_of_focus --model sscor --image_name SRS_out_of_focus.tif --offset_size 100 --repeat 1
```
- Run time: 41s (on a workstation with a NVIDIA RTX 3090 GPU)
- The corrected result will be saved in: `./raw_image/fig4/result`

## Train SSCOR
### Proximity sampling

Run sampling script to creat the training dataset.

The script will process the images in the sub-folders of `./sample_result/` folder and make a new folder `sample_{image name}` in the corresponding image folder. You can find the sampling patches in the `sample_{image name}`.

- Only horizontal or vertical stripes (e.g. SRS_stripe.tif, oblique_stripe_1.jpg, oblique_stripe_2.jpg)
```bash
python sample/sample_stripe.py --h --h_n 11 --in_dir ./raw_image/fig4 --img_name SRS_stripe.tif
python sample/sample_stripe.py --v --v_n 11 --in_dir ./raw_image/fig2 --img_name oblique_stripe_1.jpg
python sample/sample_stripe.py --v --v_n 10 --in_dir ./raw_image/fig3 --img_name oblique_stripe_2.jpg
```

- There are both horizontal and vertical stripes (e.g. MPM_stripe_2.tif, HE_1.tif, MPM_grid_1.tif)
```bash
python sample/sample_stripe_2.py --h_n 6 --v_n 6 --in_dir ./raw_image/fig3 --img_name MPM_stripe_2.tif --direction 0
python sample/sample_stripe_2.py --h_n 5 --v_n 5 --in_dir ./raw_image/fig5 --img_name HE_1.tif --direction 1 --patch_size 512
python sample/sample_stripe_2.py --h_n 9 --v_n 9 --in_dir ./raw_image/fig2 --img_name MPM_grid_1.tif --patch_size 128
```

- Stripe with out-of-focus (e.g. MPM_stripe_1.tif, SRS_out_of_focus.tif)
```bash
python sample/sample_out_of_focus.py --h --h_n 11 --in_dir ./raw_image/fig2 --img_name MPM_stripe_1.tif --x_loc 4170 --y_loc 1960
python sample/sample_out_of_focus.py --h --h_n 11 --in_dir ./raw_image/fig4 --img_name SRS_out_of_focus.tif --x_loc 2600 --y_loc 3200
```

- Stripe with bubble (e.g. SRS_bubble.tif)
```bash
python sample/sample_bubble.py --h --h_n 11 --in_dir ./raw_image/fig4 --img_name SRS_bubble.tif
```

- Scanning fringe artifact (e.g. SRS_SFA.tif)
```bash
python sample/sample_sfa.py --in_dir ./raw_image/fig4 --img_name SRS_SFA.tif --x_loc 3200
```

### Adversarial self-training

- Train a model on the sampling patches:
```bash
python train.py --dataroot ./sample_result/sample_SRS_stripe --name SRS_stripe-train --model sscor --display_id 0 --load_size 286 --crop_size 256
```
To see more intermediate results, check out `./checkpoints/`. The .pth file will be save in the corresponding folder.

## Contact
If you have any questions, please contact Shu Wang at [shu@fzu.edu.cn](shu@fzu.edu.cn) or Wenxi Liu at [wenxiliu@fzu.edu.cn](wenxiliu@fzu.edu.cn).
