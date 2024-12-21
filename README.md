# Description
- 네이버 부스트캠프 CV트랙 첫번째 프로젝트를 위한 베이스라인 코드입니다
- 주어진 baseline에서부터 시작하여 실험을 쉽게 돌릴 수 있도록 refactoring하였습니다

# 파일 구조
```
📦EDA
 ┣ 📜EDA.ipynb
 ┣ 📂_list
📦config
 ┣ 📜test_setting.yml
 ┣ 📜test_transform.yml
 ┣ 📜train_transform.yml
 ┣ 📜training_setting.yml
 ┗ 📜transform.json
📦dataloader
 ┣ 📜dataloader.py
 ┗ 📜preprocess.py
📦model
 ┣ 📜_loss.py
 ┣ 📜_model.py
 ┣ 📜_optimizer.py
 ┗ 📜_schedular.py
📦util
 ┣ 📜seed.py
 ┣ 📜utility.py
 ┗ 📜visualize.py
📜.gitignore
📜README.md
📜infer.py
📜requirements.txt
📜train.py
```


# dataloader
- preprocessor : augmentation과 전처리를 담당하는 모듈.
- dataloader : image -> tensor의 dataloader를 정의하는 모듈


# config
- train, inference시에 transform이 제대로 적용되었는지 디버깅 가능(사용한 세팅을 ---_transform.yml로 저장)
- train, inference시의 hyperparameter를 지정 가능

# infer.py
- inference시에 사용
- test_setting.yml에서 model_name, save_result_path는 필수 체크

# train.py
- train시에 사용
- train_setting.yml에서 model_name과 save_result_path 확인 필수
- wandb를 사용할 시 project_name과 test_name 확인 필수


# optimizer, loss, schedular
- yml로 쉽게 관리하기 위하여 모듈화
- loss 추가 방법은 notion 참고
