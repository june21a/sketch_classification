##############################################
# transform을 받아서 dataloader를 정의하는 모듈 #
##############################################

import cv2
import pandas as pd
import os
import torch
from typing import Tuple, Any, Callable, List, Optional, Union
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split



class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        # 데이터셋의 기본 경로, 이미지 변환 방법, 이미지 경로 및 레이블을 초기화합니다.
        self.root_dir = root_dir  # 이미지 파일들이 저장된 기본 디렉토리
        self.transform = transform  # 이미지에 적용될 변환 처리
        self.is_inference = is_inference # 추론인지 확인
        self.image_paths = info_df['image_path'].tolist()  # 이미지 파일 경로 목록
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()  # 각 이미지에 대한 레이블 목록

    def __len__(self) -> int:
        # 데이터셋의 총 이미지 수를 반환합니다.
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        # 주어진 인덱스에 해당하는 이미지를 로드하고 변환을 적용한 후, 이미지와 레이블을 반환합니다.
        img_path = os.path.join(self.root_dir, self.image_paths[index])  # 이미지 경로 조합
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환합니다.
        image = self.transform(image)  # 설정된 이미지 변환을 적용합니다.

        if self.is_inference:
            return image
        else:
            target = self.targets[index]  # 해당 이미지의 레이블
            return image, target  # 변환된 이미지와 레이블을 튜플 형태로 반환합니다.


def train_val_split(info_df, test_size):
    # 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
    train_df, val_df = train_test_split(
        info_df, 
        test_size=test_size,
        stratify=info_df['target']
    )
    return train_df, val_df



def get_dataloader(root_dir, info_df, transform, batch_size, is_inference, seedworker):
    _dataset = CustomDataset(
        root_dir=root_dir,
        info_df=info_df,
        transform=transform,
        is_inference=is_inference
    )
    
    
    if not is_inference:
        loader = DataLoader(
            _dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seedworker
        )
    else:
        loader = DataLoader(
            _dataset,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seedworker
        )
    
    return loader

