#######################################################
## 기본적인 전처리를 담당하는 모듈
#  resize, normalization, ToTensor
#######################################################
## augmentation transform을 각각 정의하고 적용 가능 
#######################################################
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from inspect import getmembers, isclass


# tuple list를 dictionary로 바꿔줍니다. 
def Convert(tup):
    di = {}
    for a, b in tup:
        di[a] = b
    return di

# Albumentation의 모든 class를 가져오는 함수입니다.
def get_augment_dict():
    return Convert(getmembers(A, isclass))


class AlbumentationsTransform:
    def __init__(self,
                 is_train = True,
                 **kwargs):
        
        # get albumentation module list
        augmentation_dict = get_augment_dict()
        
        # parsing from kwargs
        save_name = kwargs['save_name']
        select = kwargs['select']
        setting = kwargs['setting']
        
        # default transforms
        resize_transform = [A.Resize(**(setting["Resize"] if "Resize" in setting else {"height":224, "width":224}))]
        common_transforms = [
            A.Normalize(**(setting["Normalize"] if "Normalize" in setting else {})),
            ToTensorV2()
        ]
        
        # augmentation transforms
        transform_train = []
        transform_test = []
        for s in select:
            try:
                transform_train.append(augmentation_dict[s](**(setting[s] if s in setting else {})))
            except:
                raise NameError(f"augmentation {s} is not in Albumentation")

            if s == "RandomCrop":
                transform_test.append(augmentation_dict[s](**(setting[s] if s in setting else {})))
        
        
        # test의 경우 augmentation transform은 crop을 제외하고는 적용하지 않음
        if is_train:
            self.transform = A.Compose(resize_transform + transform_train + common_transforms)
        else:
            self.transform = A.Compose(resize_transform + transform_test + common_transforms)
            
        # save augmentation setting
        A.save(self.transform, os.path.join("./config", save_name), data_format='yaml')
    
    
    
    # 저장되어 있는 transform 세팅을 yml로부터 읽어오는 메서드
    def load_transform(self, path):
        self.transform = A.load(path, data_format='yaml')
        print("transform_loaded")
        
        
        
    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환

