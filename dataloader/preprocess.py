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



class AlbumentationsTransform:
    def __init__(self,
                 image_size,
                 is_train: bool = True,
                 save_name = "transform.json",
                 shift_scale_rotate_setting = {
                    "shift_limit": 0.05, 
                    "scale_limit": 0.1, 
                    "rotate_limit": 20, 
                    "p": 0.5},
                 bright_contrast_setting = {
                     "brightness_limit": 0.2,
                     "contrast_limit": 0.2,
                     "p": 0.5
                 },
                 coarse_dropout_setting = {
                    "max_holes": 5,
                    "max_height": 20,
                    "max_width": 20,
                    "p": 0.5
                 },
                 random_crop_setting = {
                     "width": 200,
                     "height": 200
                 },
                 horizontal_flip_setting = {
                     "p": 0.5
                 },
                 augmentation_table = {
                    "bright_contrast": True,
                    "coarse_dropout": True,
                    "horizontal_flip": True,
                    "random_crop": False,
                    "shift_scale_rotate": True
                }):
        
        
        resize = [A.Resize(*image_size)]
        common_transforms = [
            A.Normalize(),
            ToTensorV2()
        ]
        
        # augmentation을 이것저것 추가하기 위해 먼저 list에 담기
        # yml로 on/off 할 수 있도록 구성하려고 이렇게 했는데 아이디어가 없다..
        aug_transforms = []
        
        # train/test 사이즈는 맞아야 하기 때문에 crop을 사용할 거면 양쪽에 모두 적용
        if augmentation_table["random_crop"]:
            aug_transforms.append(A.RandomCrop(**random_crop_setting))
        
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            if augmentation_table["horizontal_flip"]:
                aug_transforms.append(A.HorizontalFlip(**horizontal_flip_setting))
            if augmentation_table["shift_scale_rotate"]:
                aug_transforms.append(A.ShiftScaleRotate(**shift_scale_rotate_setting))
            if augmentation_table["bright_contrast"]:
                augmentation_table.append(A.RandomBrightnessContrast(**bright_contrast_setting))
            if augmentation_table["coarse_dropout"]:
                aug_transforms.append(A.CoarseDropout(**coarse_dropout_setting))
            
            
            self.transform = A.Compose(
                resize + aug_transforms + common_transforms
            )
        else:
            self.transform = A.Compose(
                resize + aug_transforms +common_transforms)
            
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

