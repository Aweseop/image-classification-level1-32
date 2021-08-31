## 🦢 BoostCamp AI Tech 2nd 🦢 
### 🐧 삼식이's First P Stage Project 🐧
<br>

부스트캠프 AI Tech 2기 32조의 마스크 착용 상태 분류 프로젝트 입니다.   

<br>

#### Members🐣

- 김종현
- 김준섭
- 유관식
- 이윤영
- 조성욱
- 한태호


# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                               
---               

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`


### workspace env
- vi /etc/bash.bashrc
- export MYNAME=ks
- source /etc/bash.bashrc


### wandb login
- pip install wandb
- wandb login 
- https://wandb.ai/authorize 에서 api key 복사 후 터미널 입력


### nohup (background execution)
- chmod 711 lab/ks/nohup_train.sh   
- nohup lab/ks/nohup_train.sh 1>/dev/null 2>&1 &     