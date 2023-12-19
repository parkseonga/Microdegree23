## 💻 **딥러닝을 활용한 저혈압 및 실제 혈압 예측**

### Introduction

- 본 프로젝트는 카이스트 GSDS 기초기계학습 마이크로디그리 프로그램 내 캡스톤 과목을 기반으로 합니다.
- 본 프로젝트는 2020년에 발행된 <Deep learning models for the prediction of intraoperative hypotension> 논문에 대한 이해를 바탕으로, 논문에서 설명하는 모델을 직접 구현 및 실험 하는 데에 목적을 둡니다.

### Team Members

| <img src="https://avatars.githubusercontent.com/u/33725048?v=4" width="80"> | <img src="https://avatars.githubusercontent.com/u/70469008?v=4" width="80"> | <img src="https://avatars.githubusercontent.com/u/77106757?v=4" width="80"> | <img src="https://avatars.githubusercontent.com/u/130381077?v=4" width="80"> |
| :-------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
|                   [박성아](https://github.com/parkseonga)                   |                  [박혜나](https://github.com/hyenagatha02)                  |                     [서병석](https://github.com/76stop)                     |                    [이승호](https://github.com/sshhoo123)                    |

### Deep learning models for the prediction of intraoperative hypotension

![image1](https://github.com/parkseonga/Microdegree23/assets/70469008/f0abb0b7-8bca-4f06-9308-818feba918f2)

#### 연구 목적

- 수술 중 저혈압 상태가 장기간 유지되면, 수술 후 합병증 유발가능성이 높아짐
- 따라서 딥러닝을 통해 수술 중 실시간으로 환자의 혈압 및 저혈압 상태를 예측하기 위한 모델 개발이 꾸준히 진행되고 있음

#### 연구 데이터

- 이전까지는 동맥압 파형(ABP)을 활용하여 저혈압을 예측하는 연구가 주를 이룸
- 그러나 혈역학적 변화는 심전도 및 호흡파형과도 연관이 있기에 현재는 광혈류측정(PPG), 심전도(ECG), 호흡시 이산화탄소를 내뱉는 양(CO2)을 활용
- 또한 ABP를 활용하지 않는 비침습적인(non-invasive)측정 데이터(PPG,ECG,CO2)만을 활용하여 저혈압 및 실제혈압을 예측할 시도

#### 모델 유형

- Invasive : ABP(동맥압 파형)을 활용 / non-invasive : 그 외 PPG, ECG, CO2 만을 활용
- 1-channel : 한가지 파형만 활용 (ABP or PPG) / multi-channel : 3가지 이상의 파형을 모두 활용 (ABP or PPG + PPG,ECG,CO2)

#### 실험 유형

- classification : 수술 중 저혈압 발생 여부 판단 (0 or 1)
- regression : 환자의 실제 혈압 수치 예측

### VitalDB open dataset

서울대병원에서 시행된 6,388건의 수술에 대해 intraoperative vital signs(수술 중 생체 신호), perioperative clinical information(수술 전후 임상 정보), perioperative laboratory results(수술 전후 실험 결과) 수집

- 데이터 형태 : 500hz 고해상도 waveform / 1-7초 간격의 numeric 형태의 biosignal data
- 수집 방법 : vital recorder 활용

### Goal

PPG 데이터를 활용한 non-invasive 1-channel 모델을 통해 classification, regression 수행

### Data

#### structure

- X : 길이 3000(30초 x 100Hz)의 PPG segment
- Y : (classification) 0 or 1 / (regression) 환자 MAP
- c : 환자 번호 (학습 X)
- a : 환자 연령, 성별 등 (학습 X)
  ![image2](https://github.com/parkseonga/Microdegree23/assets/70469008/6f0ed59b-b602-4fe5-9fa1-03c1aa42aae5)

#### Preprocess

- case 1. 한 segment 내 0 이하인 값, np.nan 이 있는 경우 제외
- case 2. waveform 의 peak 탐색 후 한 segment 내 peak의 수가 ## 개 이하인 경우 제외
- case 3. segment 내 beat의 길이들의 평균을 이용하여 불규칙적인 파형들은 제외
- case 4. segment 내 beat 들의 상관관계가 0.9 미만인 경우 제외

#### final dataset number

- classification: 3,257 cases / 193,189 samples
- regression: 3,144 cases / 308,013 samples
  ![image3](https://github.com/parkseonga/Microdegree23/assets/70469008/9d956361-634b-4e04-9fb0-cd8f9adc1076)

#### train/valid/Test

- train : valid : test = 6 : 2 : 2
  ![image4](https://github.com/parkseonga/Microdegree23/assets/70469008/f905cefe-dc80-49ba-917e-47e558b8d843)

### Method

#### Data

- 입력 : 30초 x 100Hz 길이의 PPG (Photoplethysmography) 데이터
- 출력 : Hypotension for classification, MAP(평균동맥압) for regression
  - Hypotension: (MAP ≤65 mm Hg) lasting >1 min
  - Non-hypotension: (MAP > 65 mm Hg) stable for >20 min.

#### Model

a. basic CNN : 논문에서 구현한 7-layer 구성 기반
b. LSTM(long short term memory) : 시계열 데이터용 모델
c.CNN+LSTM : 이미지 데이터용 기법을 선 적용한 후 시계열 데이터용 기법을 복합 적용한 모델
d. RESNET : 대표적인 CNN 모델인 resnet을 1-dimention으로 구현한 모델 (reference : https://github.com/hsd1503/resnet1d)

#### Criteria (fixed hyperparameter)

- batch size : 128
- epoch : 100 (+ early stopping)
- Loss : (classification) BCE, (regression) L1, MSE
- Evaluation : (classification) auc, recall / (regression) mae, r2 score
- optimizer = adam / learning_rate = 1e-3 / Schedular = None

### Result

(will be added)

### Limitaions and Future works

Classfication 에서 모델 전반적으로 낮은 성능 개선 필요

- 원인 : 저혈압에 해당하는 데이터 수가 부족
  - classification : 데이터 불균형
  - regression : 예측값의 분포가 극단적인 t분포 형태를 취함
- 개선 방안
  - 데이터 불균형을 완화할 수 있는 기법 적용 (Ex label smoothing 기법, window size를 조절하여 데이터 재구성)
  - 데이터 증강 기법을 활용하여 저혈압 데이터 보충
  - 의료 신호 처리에서 많이 사용되는 모델 및 hyperparameter 탐색

## Reference

- [Deep learning models for the prediction of intraoperative hypotension 논문](https://pubmed.ncbi.nlm.nih.gov/33558051/)
- [VitalDB open dataset](https://vitaldb.net/dataset/)
- [code_repository_for_the_research 폴더](https://data.mendeley.com/datasets/wdpxsyrg2s/2)
