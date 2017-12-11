import os
import torch
import pandas as pd
from shutil import copyfile

dogs_dataframe = pd.read_csv('labels.csv')


def move_sample(phase, breed, img_id):
    directory = 'sample/{}/{}'.format(phase, breed)
    if not os.path.exists(directory):
        os.mkdir(directory)
    source = 'train_copy/{}.jpg'.format(img_id)
    destination = '{}/{}.jpg'.format(directory, img_id)

    copyfile(source, destination)

for index, row in dogs_dataframe.iterrows():
    img_id = row[0]
    breed = row[1]


    if index % 10 == 0:
        move_sample('training', breed, img_id)
    if index % 11 == 0:
        move_sample('validation', breed, img_id)
