자세히 설명 : https://blog.naver.com/jinydoggebi/224300774260

images+labels =기본 이미지(land_mark_3x3m.png) + 배경이미지(backgrounds)
windows의 wsl에서
$pip install opencv-python numpy

1단계: dataset.yaml 만들기
-----------------------------------
path: /content/dataset  # 코랩 서버가 인식할 절대 경로 (수정 불필요)
train: images
val: images

names:
  0: vertiport
-----------------------------------
  
2단계: dataset폴더(images, labels)를 압축한다. dataset.zip
- 압축은 $ zip -r dataset.zip images labels <- wsl에서
- 2개파일(dataset.yaml, dataset.zip)을 내 구글 드라이브(My Drive)에 업로드
- 코랩 노트북 열기: Google Colab에 접속하여 '새 노트'를 만듬.
- GPU 런타임 변경: 런타임>런타임유형변경> T4 GPU

3단계: 코랩 학습 코드 실행(Shift+Enter)
--------------------------------------
- from google.colab import drive
- drive.mount('/content/drive')

# 1. 코랩 내부의 /content/dataset 폴더에 압축 풀기
- !unzip /content/drive/MyDrive/yolo_dataset/dataset.zip -d /content/dataset

# 2. 데이터 구조가 잘 들어왔는지 파일 개수 확인 (검증)
import os
print("학습 이미지 개수:", len(os.listdir('/content/dataset/images')))
print("학습 라벨 개수:", len(os.listdir('/content/dataset/labels')))

- !cp /content/drive/MyDrive/yolo_dataset/dataset.yaml /content/dataset.yaml
---------------------------------------

YOLOv8 패키지 설치
----------------------------------------
- !pip install ultralytics
- 패키지 설치
----------------------------------------
------------------------------------------------------------------
from ultralytics import YOLO

# Jetson Nano/TX2에서 실시간으로 돌리기 위한 가장 가벼운 'Nano' 모델 선택
model = YOLO('yolov8n.pt') 

# 학습 실행, 착륙장 인식용 데이터셋으로 전이 학습(Transfer Learning) 시작
results = model.train(
    data='/content/dataset.yaml',  # 데이터 설정 파일 경로
    epochs=100,                    # 반복 학습 횟수 (가상 데이터는 50~100회면 충분합니다)
    imgsz=640,                     # 입력 이미지 해상도
    batch=32,                      # 한 번에 처리할 이미지 묶음 크기
    device=0                       # GPU 사용
)
------------------------------------------------------------------

4단계 : 학습완료 및 결과물 다운로드
가중치 파일 : /content/runs/detect/train/weights/best.pt
best.pi : AI 모델의 뇌

Jetson 보드에서 인공지능이 밀리지 않고 실시간(30 FPS 이상)으로 돌려면 PyTorch(.pt) 모델을 TensorRT 엔진(.engine) 포맷으로 반드시 변환

코랩에서 다운로드한 best.pt 파일을 Jetson 보드의 drone_precision_landing/ 폴더로 옮긴 후, Jetson 터미널에서 아래 명령어를 실행하여 뇌를 GPU 전용 가속 엔진으로 최적화합니다.
----------------------------------------
# Jetson 터미널에서 실행 (ultralytics가 설치되어 있어야 합니다)
python3 -m ultralytics.bin.yolo export model=best.pt format=engine imgsz=640 half=True device=0
----------------------------------------

안되면, 구글 코랩에서 미리 구워오기
!yolo export model=/content/runs/detect/train/weights/best.pt format=engine imgsz=640 half=True device=0

실행하고 나면, best.engine 생성이 안되고 best.onnx 생성, 이 친구도 된다.

변환이 성공하면 /content/runs/detect/train/weights/ 폴더 안에 best.engine 파일이 새로 생깁니다.
이 best.engine 파일을 다운로드하여 Jetson 보드의 ~/drone_precision_landing/ 폴더로 바로 옮겨주시면 끝납니다.

Jetson 터미널에서 ONNX를 TensorRT Engine으로 수동 변환하기
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

생성된 best.onnx 다운로드 방법.
-----------------------------------
from google.colab import files

# 코랩 로컬 경로에 있는 ONNX 파일을 내 컴퓨터로 강제 다운로드 시도
files.download('/content/runs/detect/train/weights/best.onnx')
-----------------------------------

구글 드라이브 yolo_dataset 폴더로 복사하기
cp /content/runs/detect/train/weights/best.onnx /content/drive/MyDrive/yolo_dataset/best.onnx


TensorRT 변환을 완성
---------------------
# Jetson 터미널에서 최종 수동 엔진 빌드
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
---------------------


최종 가동
python3 jetson_inference.py


