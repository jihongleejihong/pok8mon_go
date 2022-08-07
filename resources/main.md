
### 프로젝트 목적

- 합성곱신경망(Convolutional Neural Network, CNN)을 이용한 이미지 처리에 대한 이해
- 인공지능 서비스 개발 과정에 대한 이해 및 수행

![project preview](https://raw.githubusercontent.com/jihongleejihong/pok8mon_go/main/resources/img/main_img/1.png?raw=true){: width = "50%"}

### 프로젝트 요약


![project workflow](https://raw.githubusercontent.com/jihongleejihong/pok8mon_go/main/resources/img/main_img/2.png?raw=true)


- 본 프로젝트를 수행하며 포켓몬 149종에 관한 데이터를 활용하였음. 
    
-  6825개 이미지와 800개 Stats. 데이터를 과제 별 목적에 맞게 가공하여 모델 학습에 이용함.
    

| 과제 | Dataset 원본 | Train set | Test set | source |
| --- | --- | --- | --- | --- |
| 1. 종류 분류 | (6784, 150, 150, 3) | (5427, 150, 150, 3) | (1357, 150, 150, 3) | [이미지]  |
| 2. 속성 분류 | (6825, 150, 150, 3) | (5324, 150, 150, 3) | (1501, 150, 150, 3) | [능력치] |
| 3. Stats. 예측 | (800, 13) |  |  | [능력치] |  

#

| 과제 | 이미지 처리 기법 | 딥러닝 레이어 | Optimizer | Metrics | Loss function |
| --- | --- | --- | --- | --- | --- |
| 1. 종류 분류        2. 속성 분류 | transfer learning      (1) DenseNet201       (2) ResNet50   augmentation      - flip, rotation, shift, contrast  | Dense,    Dropout, GlobalAverage   Pooling2D 등 | Adam | accuracy | categorical_   crossentropy |
|  3. Stats. 예측 |  |  |  |  |  |



## 2. 활용 데이터

### 2.1. 데이터 출처

- Image dataset - [[Kaggle - 7000 hand-cropped and labeled Pokemon images for classification]](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)
- Stats. dataset - [[Kaggle - 721 Pokemon with stats and types]](https://www.kaggle.com/datasets/abcsds/pokemon)

### 2.2. 데이터 미리 보기

- 이미지 데이터

아래와 같이 각 포켓몬에 대해 다양한 각도, 배경, 효과가 적용된 여러 이미지를 학습에 적용함 

(포켓몬 149종 6825개 → 1마리 당 평균 46개 이미지 학습)





- Stats. 데이터

    |||
    |--|-|
    |![Untitled](https://raw.githubusercontent.com/jihongleejihong/pok8mon_go/main/resources/img/main_img/3.png?raw=true)|![Untitled](https://raw.githubusercontent.com/jihongleejihong/pok8mon_go/main/resources/img/main_img/4.png?raw=true)|
    |||

   ![Untitled](https://raw.githubusercontent.com/jihongleejihong/pok8mon_go/main/resources/img/main_img/5.png?raw=true)
    
    - [데이터 컬럼 정보]
        - **`#, Name`**: 포켓몬 별 고유 번호, 이름
        - **`Type 1, Type 2`**: 포켓몬 별 고유 속성 ⇒ 특정 공격에 대해 취약 / 방어력이 높음을 의미
        
        ---
        
        - **`Total`** : 모든 Stats. 정보(HP~Speed)의 합 ⇒ 해당 포켓몬이 얼마나 강한지를 의미
        - **`HP`**: 포켓몬이 견딜 수 있는 데미지의 총합
        - **`Attack`**: Scratch, Punch 등 일반 공격의 타격
        - **`Defense`**: 일반 공격에 대한 데미지 저항
        - **`Sp. Atk`**: Fire blast, Bubble beam 등 특정 포켓몬에 특화된 공격의 타격
        - **`Sp. Def`**: Special attack에 대한 데미지 저항
        - **`Speed`**: 포켓몬의 속도 ⇒ 선제 공격 여부 결정
        - **`Generation`**: 포켓몬 세대 (세대가 진화할 수록, 일부 Type이 추가됨)
        - **`Legendary`**: 전설의 희귀 포켓몬 여부 (bool)
    

## 3. 프로젝트 결과

### 3.1. 포켓몬 종류 분류 (Name)

![(과제 1)에 대한 문제 상황 및 학습 결과 요약](https://raw.githubusercontent.com/jihongleejihong/pok8mon_go/main/resources/img/main_img/6.png?raw=true)

(과제 1)에 대한 문제 상황 및 학습 결과 요약


**Accuracy**        = (`0.9921`)                             
**Validataion accuracy**  = (`0.8928`)



---

### 3.2. 포켓몬 속성 분류 (Type 1, Type2)

![Untitled](https://raw.githubusercontent.com/jihongleejihong/pok8mon_go/main/resources/img/main_img/7.png?raw=true)


**Accuracy**       (`Type 1`, `Type 2`) = (`0.909`, `0.938`)

**Validation accuracy** (`Type 1`, `Type 2`) = (`0.879`, `0.917`)



---
