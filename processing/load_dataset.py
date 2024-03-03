import collections
from pathlib import Path

import pandas as pd

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

collections.Iterable = collections.abc.Iterable

INPUT_HEIGHT = 160
img_dir = '../data/wiki_crop_filtered_mirrored_pose/'
BATCH_SIZE = 32


class ImageDataset(Dataset):
    """
    Class that creates dataset for training task
    Arguments:
    img_dir - directory of all images
    df - dataframe with cleaned dataset
    transform - transform
    """

    def __init__(self, img_dir, df=None, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df['image_path'][idx]
        # image = plt.imread(img_path)
        image = Image.open(img_path)
        bbox = self.df['bbox'][idx]
        image = image.crop(bbox)
        # image = Image.fromarray((image)*255)
        # image = Image.fromarray(np.uint8((image)*255))

        if image.mode != 'RGB':
            image = image.convert('RGB')
            # print(np.array(image).shape)

        if self.transform:
            # print('transforming image')
            image = self.transform(image)
            image = image.permute(1, 2, 0)
            # print(image.size())

        target = str(self.df['image_path'][idx])
        return image, target


def generate_df(image_dir):
    path = Path(image_dir)
    json_df = pd.read_json('../data/filtered_pose_mirrored.json')
    json_df = json_df.transpose()
    json_idx = json_df.index.to_list()
    data = [[p, json_df.loc[[str(p.name)]]] for p in path.glob('*') if (not p.name.endswith('mask.jpg'))
            and (p.name in json_idx)]
    image_path = [d[0] for d in data]
    bbox = [(d[1])['bbox'].values[0] for d in data]
    data = pd.DataFrame({'image_path': image_path, 'bbox' : bbox})
    return data


def generate_dataset(data_path: str | None = None):
    if data_path is None:
        data_path = img_dir

    data = generate_df(data_path)
    resize = transforms.Resize((INPUT_HEIGHT, INPUT_HEIGHT))
    base_transform = transforms.Compose(
        [resize, transforms.ToTensor()]
    )

    dataset = ImageDataset(img_dir, df=data, transform=base_transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    return data, dataset, data_loader
