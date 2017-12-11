from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

dogs_dataframe = pd.read_csv('labels.csv')


for index, row in dogs_dataframe.iterrows():
    img_id = row[0]
    breed = row[1]

    directory = 'train/{}'.format(breed)
    if not os.path.exists(directory):
        os.mkdir(directory)
    source = 'train/{}.jpg'.format(img_id)
    destination = '{}/{}.jpg'.format(directory, img_id)

    os.rename(source, destination)
