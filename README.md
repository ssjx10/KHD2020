# Korea Health Datathon 2020

Project 기간: 2020년 9월 21일 → 2020년 9월 25일

[Korea-Health-Datathon/KHD2020](https://github.com/Korea-Health-Datathon/KHD2020)

# Overview

## Korea Health Datathon 2020

- 의료 영상 데이터(부비동 X-ray 영상, 유방암 디지털 병리 영상)를 활용하여 실제 구현 가능한 인공지능 학습 모델을 기획하고 개발하는 데이터톤 대회
- NSML(Naver Smart Machine Learning)을 이용하여 학습을 진행하고, 1시간에 1번씩 모델을 제출하여 성능을 비교하는 형식
- A Track : 부비동 X-ray 이미지를 활용해 환자의 좌, 우측 상악동의 부비동염 여부를 분류해내는 딥러닝 모델 개발
- B Track : 양성과 악성으로 구성된 병리 이미지들을 효율적으로 분류하는 딥러닝 모델 개발
- 이 중 B Track에 참여하였고, 2등상 수상

## Data (B Track)

Train Data의 수 : 양성 5,600장 / 음성 2,400장

Test Data의 수 : 양성 1,400장 / 음성 600장

![Korea%20Health%20Datathon%202020%20a6a400c4fabc4f41accae5245596f762/Untitled.png](image/Untitled.png)

# 평가 지표

Accuracy, Specificity, Sensitivity, Precision (PPV), Negative predictable value (NPV), F1 score의 평균값

# Method

- ImageNet Data로 pretrain 되어진 VGG, ResNet, DenseNet, InceptionNet을 Fine tuning을 해보며 성능을 비교하였고, 데이터의 수가 엄청 많지 않아서인지 ResNet50이 가장 좋은 성능을 보였다.
- Fine tuning을 할 때 ImageNet 데이터의 사이즈보다 큰 사이즈로 학습했을 때 성능이 좋았다.
- Kaggle의 Breast Histopathology Images Dataset을 이용하여 ResNet과 EfficientNet을 학습시킨 후 Fine tuning을 해보았지만, 위의 ImageNet으로 학습시킨 ResNet이 더 좋은 성능을 보였다.
- Augmentation을 해주기 위해 아래 두 논문을 참고 하였지만, 최종적으로 RandomHorizontalFlip과 RandomVerticalFlip만을 적용하였을 때가 가장 좋은 성능을 보였다.

[](https://arxiv.org/pdf/2007.05008.pdf)

[](https://arxiv.org/pdf/1902.06543.pdf)

- 이외에도 Ensemble, Knowledge Distillation등 성능을 높히기 위해 다양한 실험을 적용해 주었다.
- Train과 Validation으로 나누어서 학습을 진행하였지만, Train set으로 Overfitting 해주는 것이 성능 향상에 더 좋을 듯하여 나누어 주었던 Validation set을 다시 합쳐서 학습 시켰고, 이때 성능의 향상을 보였다.

# Code

main.py
- 5 Fold CV

dummy_main.py
- split validation or full train
ensemble_main.py
- inference in ensemble model

ensemble2_main.py
- train a student models
ex) DML

customEval.py
- Accuracy, Specificity, Sensitivity, Precision, Negative Predictable value, F1 score

nsml command
- nsml run -d Breast_Pathology -e main.py --shm-size 16G
- nsml model ls -j KHD005/Breast_Pathology/72
- nsml submit KHD005/Breast_Pathology/72 6
