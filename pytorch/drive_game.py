#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:06:12 2017
@author: leoara01
Examples:
python3 drive_on_game.py --ip=10.45.64.32 --model=./cnn_14.pkl

References:
https://discuss.pytorch.org/t/how-to-cast-a-tensor-to-another-type/2713/4
https://github.com/pytorch/examples/issues/134
https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py
"""
import fire
import argparse
import game_communication
import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.misc
from model import CNNDriver
import time

# Force to see just the first GPU
# https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
import os

class GameRecord:
    __m_id = 0
    __m_img = 0
    __m_telemetry = []

    def __init__(self, id_record, img, telemetry):
        self.__m_id = id_record
        self.__m_img = img
        self.__m_telemetry = telemetry

    def get_id(self):
        return self.__m_id

    def get_image(self):
        return self.__m_img

    def get_telemetry(self):
        return self.__m_telemetry


class Drive:
    __device = []
    __model = []
    def __init__(self):
        self.__device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def drive(self, ip='127.0.0.1', port=50007, model='cnn_18.pkl', crop_start=126, crop_end=226):
        print("Loading model: %s" % model)
        cnn = CNNDriver()
        # Model file trained with gpu need to be remaped on CPU
        if self.__device.type == 'cpu':
            cnn.load_state_dict(torch.load(model, map_location='cpu'))
        else:
            cnn.load_state_dict(torch.load(model))
        cnn.eval()
        cnn = cnn.to(self.__device)

        transformations = transforms.Compose([transforms.ToTensor()])
        comm = game_communication.GameTelemetry(ip, port)
        comm.connect()

        # Run until Crtl-C
        try:
            list_records = []
            degrees = 0
            while True:
                # Sleep for 50ms
                time.sleep(0.05)

                # Get telemetry and image
                telemetry = comm.get_game_data()
                cam_img = comm.get_image()

                # Skip entire record if image is invalid
                if (cam_img is None) or (telemetry is None):
                    continue

                start = time.time()
                # Resize image to the format expected by the model
                cam_img_res = (scipy.misc.imresize(np.array(cam_img)[crop_start:crop_end], [66, 200]))
                torch_tensor = transformations(cam_img_res)
                cam_img_res = torch_tensor.unsqueeze(0)
                cam_img_res = cam_img_res.to(self.__device)

                # Get steering angle from model
                degrees = cnn(cam_img_res)

                # Convert Variable to numpy
                degrees = float(degrees.data.cpu().numpy())
                end = time.time()
                elapsed_seconds = float("%.2f" % (end - start))
                print('Elapsed time:', elapsed_seconds, 'angle:', degrees)


                # Send command to game here...
                commands = [degrees, 0.5]
                comm.send_command(commands)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    # Call function that implement the auto-pilot
    fire.Fire(Drive)