import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
import torch
from tqdm.auto import tqdm


class ImageVisualization():
    def __init__(self, image_dir, image_info_file):
        """경로를 통해 이미지를 읽어오기 위한 dataframe 구축
        Args:
            image_dir (str): 이미지들이 위치한 가장 상위 폴더 "./data/train"
            image_info_file (str): 이미지 메타 데이터 csv파일 위치
        """
        self.df = pd.read_csv(image_info_file)
        self.images = list(map(lambda x : x.replace("\\", "/"), glob(image_dir + "/*/*")))
        
        self.image_prop = defaultdict(list)

        progress_bar = tqdm(enumerate(self.images), desc="extracting", leave=False)
        for i, path in progress_bar:
            with Image.open(path) as img:
                self.image_prop['height'].append(img.height)
                self.image_prop['width'].append(img.width)
                self.image_prop['img_aspect_ratio'] = img.width / img.height
                self.image_prop['mode'].append(img.mode)
                self.image_prop['format'].append(img.format)
                self.image_prop['size'].append(round(os.path.getsize(path) / 1e6, 2))
            self.image_prop['path'].append(path)
            self.image_prop['image_path'].append(path.split('/')[-2] + "/" + path.split('/')[-1])

        image_data = pd.DataFrame(self.image_prop)
        self.image_data = image_data.merge(self.df, on='image_path')
    
    
    def show_class_images(self, target):
        """target class의 이미지를 모두 보여줌

        Args:
            target (int): target class의 번호
        """
        len_data = len(self.image_data[self.image_data['target'] == target])
        fig, axs = plt.subplots((len_data // 5)+1, 5, figsize=(16, 10))
        images = self.image_data[self.image_data['target'] == target]['path'].values
        for i, path in enumerate(images):
            img = Image.open(path)
            ax = axs[i // 5, i % 5]  # Use double indexing for 2D subplots
            ax.imshow(img)
            ax.axis('off')
        plt.show()
    
    
    def augmentation_compare(self, target, transform):
        """ 변환된 이미지와 원본 이미지 비교를 위한 시각화 함수

        Args:
            target (int): target class의 번호
            transform (A.Compose()): Albumentation Compose class
        """
        self.show_class_images(target)
        len_data = len(self.image_data[self.image_data['target'] == target])
        fig, axs = plt.subplots((len_data // 5)+1, 5, figsize=(16, 10))
        images = self.image_data[self.image_data['target'] == target]['path'].values
        for i, path in enumerate(images):
            image = cv2.imread(path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).detach().permute(1, 2, 0).clip(0, 1)
            ax = axs[i // 5, i % 5]  # Use double indexing for 2D subplots
            ax.imshow(image)
            ax.axis('off')
        plt.show()


if __name__ == "__main__":
    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    test = ImageVisualization(traindata_dir, traindata_info_file)
    