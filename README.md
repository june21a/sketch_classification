# 파일 구조

|-- data ( data folder )
|   |-- sample_submission.csv
|   |-- test
|   |-- test.csv
|   |-- train
|   `-- train.csv
|-- june21a ( project folder )
|   |-- baseline_code
|   |-- config
|   |-- dataloader
|   |-- infer.py
|   |-- model
|   |-- requirements.txt
|   |-- timm_list.txt
|   |-- train.py
|   |-- util



# dataloader
- preprocessor : augmentation과 전처리를 담당하는 모듈.
- dataloader : image -> tensor의 dataloader를 정의하는 모듈


# config
- train, inference시에 transform이나 setting에 대해 저장하는 yml파일이 위치합니다
- train, inference전에 세팅은 여기서 바꾸시면 됩니다
- 나중에 많이 바꿔가며 실험해야할 부분은 argparser로 대체할 예정

# infer.py
- inference시에 사용
- test_setting.yml에서 model_name, save_result_path는 꼭 바꾸도록 해주세요
- ensemble은 아직 구현하지 않았습니다.

# train.py
- train시에 사용
- augmentation과 훈련 설정을 여기서 바꿀 수 있습니다
- 새로운 augmentation을 더 적용하려면 바꿔야하는 부분은 주석에 설명되어 있습니다
- 마찬가지로 훈련 전 model_name과 save_result_path는 꼭 바꾸도록 해주세요
- wandb를 사용할 시 project_name과 test_name도 정확한 기록을 위해 주의해 주세요


# optimizer, loss, schedular
- 각각에 대한 .py파일이 model폴더에 존재하며 불러오기 좀 더 쉽게 모듈화를 하려고 했지만 큰 의미는 없는 것 같습니다
- 다른 optimizer, logg, schedular를 사용하고 싶으시면 train.py의 main()함수에서 간단히 수정 가능합니다.
