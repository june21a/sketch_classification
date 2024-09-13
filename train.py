import os
import pandas as pd
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from model import _loss, _model, _optimizer, _schedular
from util import seed
from dataloader import dataloader, preprocess
import wandb
import matplotlib.pyplot as plt


# 원본 baseline 코드에서 가져온 class에 몇가지 메서드가 추가되었습니다.
class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model.to(device)  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    
    # epoch이 진행되면서 acc, loss의 양상을 보여줍니다
    def plot_metrics(self):
        epochs = range(1, self.epochs + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'metrics.png'))
        plt.show()
    
    
    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")


    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        total_num = 0
        total_correct = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            
            # acc 계산
            with torch.no_grad():
                outputs = torch.argmax(outputs, axis=1)
                total_correct += torch.sum(outputs == targets).item()
                total_num += images.shape[0]
            progress_bar.set_postfix(loss=loss.item(), acc=total_correct / total_num)
        
        
        final_loss = total_loss / len(self.train_loader)
        final_acc = total_correct / total_num
        # 1에폭 끝나면 wandb로 로깅
        wandb.log({'training_loss' : final_loss,
                   'training_acc' : final_acc})
        
        self.train_losses.append(final_loss)
        self.train_accuracies.append(final_acc)
        return final_loss, final_acc

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        
        total_loss = 0.0
        total_num = 0
        total_correct = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                outputs = torch.argmax(outputs, axis=1)
                total_correct += torch.sum(outputs == targets).item()
                total_num += outputs.shape[0]
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), acc=total_correct / total_num)
        
        final_loss = total_loss / len(self.val_loader)
        final_acc = total_correct / total_num
        wandb.log({'validation_loss' : final_loss,
                   'validation_acc' : final_acc})
        
        self.val_accuracies.append(final_acc)
        self.val_losses.append(final_loss)
        return final_loss, final_acc


    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Validation Loss: {val_loss:.4f} , Validation acc: {val_acc:.4f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step()
        
        self.plot_metrics()





def main():
    ##########################################################################################################
    ########## setting. 나중에 argparser로 대체할 예정. infer.py도 마찬가지
    # 만약 argparser로 할거면 augmentation setting은 고정해두고 해야할듯
    with open('./config/training_setting.yml', 'r') as f:
        config = yaml.full_load(f)
    
    
    # parse directory setting
    os.chdir(config["project_dir"])
    traindata_dir = config["traindata_dir"]
    traindata_info_file = config["traindata_info_file"]
    
    
    # parse training setting
    TEST_SIZE = config["test_size"]
    BATCH_SIZE = config["batch_size"]
    EPOCHS = config["epochs"]
    save_result_path = config["save_result_path"]
    
    SCHEDULAR_TYPE = config["schedular_type"]
    OPTIMIZER = config["optimizer"]
    LEARNING_RATE = config["learning_rate"]
    WEIGHT_DECAY = config["weight_decay"]
    
    
    # parse model info
    MODEL_TYPE = config["model_type"]
    MODEL_NAME = config["model_name"]
    IS_PRETRAINED = config["is_pretrained"]
    

    # parse augmentation and preprocessing setting
    IMAGE_SIZE = config["image_size"]
    shift_scale_rotate_setting = config["augmentation"]["shift_scale_rotate"]
    bright_contrast_setting = config["augmentation"]["bright_contrast"]
    coarse_dropout_setting = config["augmentation"]["coarse_dropout"]
    random_crop_setting = config["augmentation"]["random_crop"]
    horizontal_flip_setting = config["augmentation"]["horizontal_flip"]
    augmentation_table = config["augmentation"]["augmentation_table"]
    ##########################################################################################################
    
    ##########################################################################################################
    ##### wandb setting
    
    wandb.init(project=config["project_name"])
    wandb.run.name = config["test_name"]
    wandb.run.save()
    
    wandb.config.update(config)
    ##########################################################################################################
    
    # device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # seed
    seed.seed_everything(42)
    
    # load image meta data csv
    df = pd.read_csv(traindata_info_file)
    num_classes = len(df["class_name"].unique())
    
    # validation data 나누기
    train_df, test_df = dataloader.train_val_split(df, TEST_SIZE)
    
    
    # train, test image에 대한 전처리 정의
    train_transform = preprocess.AlbumentationsTransform(image_size = IMAGE_SIZE, 
                                                         is_train=True,
                                                         save_name="train_transform.yml",
                                                         shift_scale_rotate_setting = shift_scale_rotate_setting,
                                                         bright_contrast_setting = bright_contrast_setting,
                                                         coarse_dropout_setting = coarse_dropout_setting,
                                                         random_crop_setting = random_crop_setting,
                                                         horizontal_flip_setting = horizontal_flip_setting,
                                                         augmentation_table = augmentation_table
                                                         )
    
    test_transform = preprocess.AlbumentationsTransform(image_size = IMAGE_SIZE, 
                                                         is_train=False,
                                                         save_name="test_transform.yml",
                                                         random_crop_setting = random_crop_setting,
                                                         augmentation_table = augmentation_table
                                                        )
    
    
    # train, test dataloader
    train_dataloader = dataloader.get_dataloader(root_dir = traindata_dir, 
                                                 info_df = train_df, 
                                                 transform = train_transform, 
                                                 batch_size = BATCH_SIZE, 
                                                 is_inference = False, 
                                                 seedworker = seed.seed_worker)
    
    test_dataloader = dataloader.get_dataloader(root_dir = traindata_dir, 
                                                 info_df = test_df, 
                                                 transform = test_transform, 
                                                 batch_size = BATCH_SIZE, 
                                                 is_inference = False, 
                                                 seedworker = seed.seed_worker)
    
    
    # model 정의하기
    model = _model.ModelSelector(
        model_type = MODEL_TYPE,
        num_classes = num_classes,
        model_name = MODEL_NAME,
        pretrained = IS_PRETRAINED
    ).get_model()
    
    
    # optimizer, loss, schedular
    optimizer = _optimizer.get_optimizer(OPTIMIZER, model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    criterion = _loss.get_loss()
    schedular = _schedular.get_schedular(SCHEDULAR_TYPE, optimizer, len(train_dataloader) // 64 + 1)
    
    
    # Trainer class
    trainer = Trainer(
        model = model,
        device = device,
        train_loader = train_dataloader,
        val_loader = test_dataloader,
        optimizer = optimizer,
        scheduler = schedular,
        loss_fn = criterion,
        epochs = EPOCHS,
        result_path = save_result_path
    )
    trainer.train()


if __name__ == "__main__":
    main()