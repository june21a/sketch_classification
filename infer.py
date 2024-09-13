import os
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm

from dataloader import preprocess, dataloader
from util import seed
from model import _model



# 모델 추론을 위한 함수
def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)
            
            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions


def main():
    ##############################################################
    with open("./config/test_setting.yml", "r") as file:
        config = yaml.safe_load(file)

    with open("./config/train_setting.yml", "r") as file:
        train_config = yaml.safe_load(file)
    
    ############# test setting 이것도 나중에 파일이나 argparser로 가져오면 좋을듯
    testdata_dir = config["testdata_dir"]
    testdata_info_file = config["testdata_info_file"]
    num_classes = config["num_classes"]
    
    # inference setting
    IMAGE_SIZE = train_config["image_size"]
    BATCH_SIZE = config["batch_size"]
    
    
    # model setting
    MODEL_TYPE = config["model_type"]
    MODEL_NAME = config["model_name"]
    save_result_path = config["save_result_path"]
        
    
    # Ensemble 할 시 여러 best model 불러오기
    MODEL_TYPES = config["model_types"]
    MODELS = config["models"]
    save_result_paths = config["save_result_paths"]
    ##############################################################
    
    # device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    # metadata df load
    test_info = pd.read_csv(testdata_info_file)
    
    # test image preprocessing
    test_transform = preprocess.AlbumentationsTransform(image_size = IMAGE_SIZE, 
                                                         is_train=False,
                                                         augmentation_table=train_config["augmentation"]["augmentation_table"]
                                                        )
    
    
    # test dataloader
    test_dataloader = dataloader.get_dataloader(root_dir = testdata_dir, 
                                                 info_df = test_info, 
                                                 transform = test_transform, 
                                                 batch_size = BATCH_SIZE, 
                                                 is_inference = True, 
                                                 seedworker = seed.seed_worker)
    
    
    
    # load best model
    model_selector = _model.ModelSelector(
        model_type= MODEL_TYPE, 
        num_classes=num_classes,
        model_name=MODEL_NAME, 
        pretrained=False
    )
    model = model_selector.get_model()
    
    model.load_state_dict(
        torch.load(
            os.path.join(save_result_path, "best_model.pt"),
            map_location='cpu',
            weights_only=True
        )
    )
    
    # inference
    predictions = inference(
        model=model, 
        device=device, 
        test_loader=test_dataloader
    )
    
    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info
    
    # DataFrame 저장
    test_info.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()