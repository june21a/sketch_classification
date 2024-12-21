# Description
- 네이버 부스트캠프 CV트랙 첫번째 프로젝트를 위한 베이스라인 코드입니다
- 주어진 baseline에서부터 시작하여 실험을 쉽게 돌릴 수 있도록 refactoring하였습니다

## 🗒️ 프로젝트 개요
Sketch이미지 분류 경진대회는 주어진 데이터를 활용하여 모델을 제작하고 어떤 객체를 나타내는지 분류하는 대회입니다.

Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 이 중, 비정형 데이터의 정확한 인식과 분류는 여전히 해결해야 할 주요 과제로 자리잡고 있습니다. 특히 사진과 같은 일반 이미지 데이터에 기반하여 발전을 이루어나아가고 있습니다.

하지만 일상의 사진과 다르게 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.

이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다. 이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.

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

# 사용 방법
### installation
```
git clone https://github.com/june21a/sketch_classification.git
cd sketch_classification
pip install -r requirements.txt
```

- 각각 train_setting.yml, test_setting.yml을 수정하신 후 train.py, infer.py를 실행
  ```
  python train.py
  python infer.py
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
