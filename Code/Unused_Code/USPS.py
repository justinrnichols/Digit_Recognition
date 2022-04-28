from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, Optional, Tuple
import gzip
import numpy as np
import os
import os.path
import torch
import warnings


class USPS(VisionDataset):
    # resources = [('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz',0),
    #              ('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz',1)]
    resources = [('Data/usps_train.gz', 0),
                 ('Data/usps_test.gz', 1)]

    train_file = 'train.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(USPS, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        if self.train:
            data_file = self.train_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder, self.train_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self) -> None:
        if self._check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder)
        print('Processing...')
        train_set = read_datafile(os.path.join(self.raw_folder, 'usps_train.gz'))
        test_set = read_datafile(os.path.join(self.raw_folder, 'usps_test.gz'))
        with open(os.path.join(self.processed_folder, self.train_file), 'wb') as f:
            torch.save(train_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

def read_datafile(path):
    labels, images = [], []
    with gzip.GzipFile(path) as f:
        for line in f:
            vals = line.strip().split()
            labels.append(float(vals[0]))
            images.append([float(val) for val in vals[1:]])
    labels = torch.from_numpy(np.array(labels, dtype=np.int32))
    # labels[labels == 10] = 0
    images = torch.from_numpy(np.array(images, dtype=np.float32).reshape(-1, 16, 16))
    return images, labels.long()