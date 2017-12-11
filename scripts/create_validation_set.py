import os
import torch
import pandas as pd

dogs_dataframe = pd.read_csv('labels.csv')


for index, row in dogs_dataframe.iterrows():
    img_id = row[0]
    breed = row[1]

    directory = 'valid/{}'.format(breed)
    if index % 10 == 0:
        if not os.path.exists(directory):
            os.mkdir(directory)
        source = 'train/{}/{}.jpg'.format(breed, img_id)
        destination = 'valid/{}/{}.jpg'.format(breed, img_id)

        os.rename(source, destination)
