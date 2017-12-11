import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
from train_model import data_transforms
from data.labels import get_labels

class TestImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        identifier = path.split('/')[-1].split('.')[0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, identifier

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

data_transforms = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

def predict(trained_model, directory):
    test_set = TestImageFolder(directory, data_transforms)
    test_dataloder = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=4
    )

    index = 0
    predictions = pd.DataFrame(columns=get_labels())
    for data in test_dataloder:
        images, ids = data

        if torch.cuda.is_available():
            inputs = Variable(images.cuda())
        else:
            inputs = Variable(inputs)

        outputs = F.softmax(trained_model(inputs)).data.cpu().numpy()
        ids_and_outputs = pd.DataFrame(np.c_[np.asarray(ids), outputs], columns=get_labels())

        predictions = pd.concat([predictions, ids_and_outputs], ignore_index=True)

        if index % 50 == 0:
            print('{} batches complete'.format(index))
        index += 1

    with open('predictions.csv', 'w') as f:
        f.write(predictions.to_csv(index=False))

model = torch.load('model.pth')
predict(model, 'data/test')
