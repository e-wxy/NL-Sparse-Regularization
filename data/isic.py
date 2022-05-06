import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import pandas as pd
from .utils import noisify


class Annotation(object):
    """ annotate ISIC 2018

    Attributes:
        df(pd.DataFrame): df.columns=['image_id', 'label']
        categories(list): dermatological types
        class_dict(dict): class name -> index
        label_dict(dict): index -> class name
        class_num(int): the number of classes
        
    Usages:
        count_samples(): get numbers of samples in each class

    """
    def __init__(self, ann_file: str) -> None:
        """
        Args:
            ann_file (str): csv file path
        """
        self.df = pd.read_csv(ann_file, header=0)
        self.categories = list(self.df.columns)
        self.categories.pop(0)
        self.class_num = len(self.categories)
        self.class_dict, self.label_dict = self._make_dicts()
        self.df = self._relabel()

    def _make_dicts(self):
        """ make class and label dict from categories' names """
        class_dict = {}
        label_dict = {}
        for i, name in enumerate(self.categories):
            class_dict[name] = i
            label_dict[i] = name

        return class_dict, label_dict

    def _relabel(self) -> pd.DataFrame:
        self.df.rename(columns={'image': 'image_id'}, inplace=True)
        self.df['label'] = self.df.select_dtypes(['number']).idxmax(axis=1)
        self.df['label'] = self.df['label'].apply(lambda x: self.class_dict[x])
        for name in self.categories:
            del self.df[name]
        return self.df

    def count_samples(self) -> list:
        """ count sample_nums """
        value_counts = self.df.iloc[:, 1].value_counts()
        class_nums = [value_counts[i] for i in range(len(value_counts))]
        return class_nums

    def to_names(self, nums):
        """ convert a goup of indices to string names 
        
        Args:
            nums(torch.Tensor): a list of number labels

        Return:
            a list of dermatological names
        
        """
        names = [self.label_dict[int(num)] for num in nums]
        return names



class ISIC2018(data.Dataset):
    def __init__(self, root, train: bool = True,
                 transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, random_state=123):
        self.root = root
        self.img_dir = 'ISIC2018_Task3_Training_Input'
        train_df = Annotation(os.path.join(root, 'Train_GroundTruth.csv')).df
        test_df = Annotation(os.path.join(root, 'Test_GroundTruth.csv')).df
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'isic18'
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.nb_classes = 7

        self.train_labels = np.array(list(train_df['label']))
        self.train_img_ids = list(train_df['image_id'])
        self.test_labels = np.array(list(test_df['label']))
        self.test_img_ids = list(test_df['image_id'])

        self.train_noisy_labels, self.actual_noise_rate = noisify(self.dataset, self.nb_classes, self.train_labels, 'symmetric', self.noise_rate, random_state)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type != 'clean':
                img_id, target = self.train_img_ids[index], self.train_noisy_labels[index]
            else:
                img_id, target = self.train_img_ids[index], self.train_labels[index]
        else:
            img_id, target = self.test_img_ids[index], self.test_labels[index]


        pth_img = os.path.join(self.root, self.img_dir, img_id + '.jpg')
        img = Image.open(pth_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)


