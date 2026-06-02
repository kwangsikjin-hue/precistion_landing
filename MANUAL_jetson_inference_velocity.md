# Jetson 정밀 착륙 비전 시스템 매뉴얼

**파일명:** `jetson_inference_velocity.py`  
**버전:** v2.0 (칼만 필터 + 속도 추정 적용)  
**작성일:** 2026-06-03  
**대상 하드웨어:** NVIDIA Jetson Nano + Intel RealSense D435i + Pixhawk (ArduPilot)

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [하드웨어 구성](#2-하드웨어-구성)
3. [소프트웨어 의존성](#3-소프트웨어-의존성)
4. [시스템 동작 구조](#4-시스템-동작-구조)
5. [모듈별 상세 설명](#5-모듈별-상세-설명)
   - 5.1 GStreamer 영상 스트리밍
   - 5.2 MAVLink 비행 제어기 연결
   - 5.3 TensorRT 추론 엔진 (JetsonTRTEngine)
   - 5.4 칼만 필터 추적기 (KalmanFilter3D)
   - 5.5 RealSense 깊이 처리
   - 5.6 메인 처리 루프
6. [좌표계 변환 흐름](#6-좌표계-변환-흐름)
7. [화면 출력 설명](#7-화면-출력-설명)
8. [주요 설정값 튜닝 가이드](#8-주요-설정값-튜닝-가이드)
9. [실행 방법](#9-실행-방법)
10. [오류 대처](#10-오류-대처)

---

## 1. 시스템 개요

본 소프트웨어는 **드론의 자율 정밀 착륙(Precision Landing)** 을 위한 엣지 비전 시스템입니다.

Jetson Nano 상에서 실행되며, 다음 세 가지 핵심 기능을 수행합니다.

| 기능 | 설명 |
|------|------|
| **착륙 패드 탐지** | YOLOv8 AI 모델을 TensorRT로 가속해 실시간 객체 탐지 |
| **위치·속도 추정** | RealSense D435i 깊이 카메라 + 칼만 필터로 3D 위치 및 이동 속도 계산 |
| **FC 착륙 유도** | MAVLink LANDING_TARGET 메시지를 Pixhawk에 송신해 자동 착륙 유도 |

추가로 처리된 영상을 GStreamer UDP 스트림으로 지상국(GCS)에 실시간 전송합니다.

---

## 2. 하드웨어 구성

```
┌──────────────────────────────────────────────────────┐
│                    Jetson Nano                       │
│                                                      │
│  ┌─────────────────┐    ┌──────────────────────────┐ │
│  │ RealSense D435i │    │   TensorRT GPU 추론       │ │
│  │ - 컬러: 640×480 │───►│   YOLOv8 best.engine     │ │
│  │ - 깊이: 640×480 │    └──────────┬───────────────┘ │
│  │ - 30fps         │               │                 │
│  └─────────────────┘               ▼                 │
│                            ┌───────────────┐         │
│                            │ 칼만 필터     │         │
│                            │ 위치+속도 추정│         │
│                            └──────┬────────┘         │
│                                   │                  │
│              ┌────────────────────┼──────────────┐   │
│              ▼                    ▼              ▼   │
│     ┌──────────────┐   ┌──────────────┐  ┌────────┐ │
│     │ MAVLink UDP  │   │ GStreamer UDP │  │ 화면   │ │
│     │ → Pixhawk FC │   │ → GCS PC     │  │ 출력   │ │
│     └──────────────┘   └──────────────┘  └────────┘ │
└──────────────────────────────────────────────────────┘
```

### 연결 구성

| 장치 | 연결 방식 | 주소/포트 |
|------|-----------|-----------|
| Pixhawk FC | UDP (MAVLink) | `udp:127.0.0.1:14551` |
| GCS PC (영상) | UDP (GStreamer) | `192.168.1.30:5600` |
| RealSense D435i | USB 3.0 | - |

> **참고:** Pixhawk를 USB-UART 모듈로 직접 연결하는 경우  
> `mavutil.mavlink_connection('/dev/ttyUSB0', baud=921600)` 으로 변경

---

## 3. 소프트웨어 의존성

| 라이브러리 | 역할 |
|-----------|------|
| `tensorrt` | YOLOv8 TensorRT 추론 엔진 |
| `pyrealsense2` | RealSense D435i 깊이/컬러 스트림 |
| `opencv-python` | 영상 처리, 칼만 필터, GStreamer 출력 |
| `numpy` | 행렬 연산 |
| `pymavlink` | MAVLink 프로토콜 통신 |
| `CUDA Runtime` | GPU 메모리 제어 (`/usr/local/cuda/lib64/libcudart.so`) |

### 필요 파일

| 파일 | 설명 |
|------|------|
| `best.engine` | YOLOv8 TensorRT 변환 모델 (같은 디렉토리에 위치해야 함) |

---

## 4. 시스템 동작 구조

프로그램은 시작 시 초기화를 완료한 뒤, **매 프레임(약 30fps)** 다음 순서로 동작합니다.

```
[시작]
  │
  ├─ GStreamer VideoWriter 초기화
  ├─ MAVLink Pixhawk 연결
  ├─ TensorRT 엔진 로드 (best.engine)
  ├─ RealSense D435i 파이프라인 시작
  ├─ 깊이 필터 초기화 (spatial / temporal / hole_fill)
  ├─ 카메라 내부 파라미터(intrinsics) 1회 취득
  └─ 칼만 필터 인스턴스 생성
  │
  ▼
[메인 루프 — 30fps]
  │
  ├── ① RealSense 프레임 수신
  ├── ② 깊이-컬러 픽셀 정렬
  ├── ③ 깊이 후처리 (spatial → temporal → hole_fill)
  ├── ④ TensorRT YOLOv8 추론
  ├── ⑤ 최고 신뢰도 객체 1개 선택 (score > 0.6)
  │
  ├── [객체 감지 성공]
  │     ├── ⑥ 바운딩박스 좌표 복원 (640×640 → 640×480)
  │     ├── ⑦ 5×5 영역 중앙값 깊이 측정
  │     ├── ⑧ 2D 픽셀 → 3D 미터 좌표 변환 (역투영)
  │     ├── ⑨ 칼만 필터: 위치 보정 + 속도 추정
  │     ├── ⑩ 화면 오버레이 (위치/속도/속력 표시)
  │     └── ⑪ MAVLink LANDING_TARGET 송신 (10Hz)
  │
  ├── [객체 미감지]
  │     └── 칼만 필터 초기화 (reset)
  │
  ├── GStreamer 영상 스트리밍
  ├── 화면 표시 (imshow)
  └── 'q' 키 → 종료
  │
[종료]
  ├─ RealSense 파이프라인 정지
  ├─ GStreamer VideoWriter 해제 (out.release)
  └─ 모든 윈도우 닫기
```

---

## 5. 모듈별 상세 설명

### 5.1 GStreamer 영상 스트리밍

처리된 영상 프레임(바운딩박스·텍스트 오버레이 포함)을 실시간으로 지상국에 전송합니다.

```
파이프라인 구성:
  appsrc → videoconvert → I420 변환 → x264 인코딩 → RTP 패킹 → UDP 전송
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 해상도 | 640×480 | 원본 컬러 해상도와 동일 |
| 코덱 | H.264 (x264enc) | 압축 전송 |
| 비트레이트 | 500 kbps | 네트워크 대역폭 조절 가능 |
| 지연 설정 | `zerolatency` | 실시간 FPV용 최소 지연 |
| 인코딩 속도 | `superfast` | Jetson CPU 부하 최소화 |
| 대상 IP | `192.168.1.30:5600` | GCS PC 주소 (변경 필요) |

> **GCS 수신 방법 (QGroundControl / GStreamer):**
> ```
> gst-launch-1.0 udpsrc port=5600 ! application/x-rtp,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink
> ```

---

### 5.2 MAVLink 비행 제어기 연결

ArduPilot Pixhawk FC와 MAVLink 프로토콜로 통신합니다.

**연결 흐름:**
```
1. mavlink_connection('udp:127.0.0.1:14551') — 연결 소켓 생성
2. wait_heartbeat() — FC 생존 확인 (응답 올 때까지 대기)
3. 연결 성공 → master 객체로 메시지 송신
4. 연결 실패 → master = None (나머지 기능은 정상 동작)
```

**LANDING_TARGET 메시지 구성:**

| 필드 | 값 | 설명 |
|------|----|------|
| timestamp | `time.time() × 1e6` | 마이크로초 단위 타임스탬프 |
| target_num | `0` | 착륙 타겟 번호 (0번 고정) |
| frame | `MAV_FRAME_BODY_NED` | 기체 기준 NED 좌표계 |
| angle_x | `atan2(fx, fz)` | 가로 오프셋 각도 (라디안) |
| angle_y | `atan2(fy, fz)` | 세로 오프셋 각도 (라디안) |
| distance | `fz` | 칼만 보정된 수직 고도 (미터) |

**송신 주기:** 10Hz (30fps 전체 송신 시 FC MAVLink 버스 과부하 방지)

---

### 5.3 TensorRT 추론 엔진 (JetsonTRTEngine)

YOLOv8 모델을 Jetson GPU에서 최대 속도로 실행하는 추론 엔진입니다.

#### 초기화 (`__init__`)

```
best.engine 로드
    → TensorRT Runtime으로 역직렬화
    → 실행 컨텍스트 생성
    → 각 바인딩에 CUDA GPU 메모리 할당 (cudaMalloc)
```

| 버퍼 | 형상 | 설명 |
|------|------|------|
| 입력 | `(1, 3, 640, 640)` | 배치1, RGB 채널, 640×640 |
| 출력 | `(1, 5, 8400)` | 8400개 앵커 × [x, y, w, h, score] |

#### 추론 (`infer`)

```
입력 이미지 (640×480 BGR)
    ↓ cv2.resize
640×640 리사이즈
    ↓ transpose(2,0,1)
CHW 채널 순서 변환 (OpenCV BGR→모델 RGB 순서)
    ↓ / 255.0
0.0~1.0 정규화
    ↓ cudaMemcpy (Host→Device)
GPU VRAM으로 전송
    ↓ execute_v2
GPU 추론 실행
    ↓ cudaMemcpy (Device→Host)
결과를 CPU RAM으로 회수
    ↓ 반환
(1, 5, 8400) 배열
```

#### 메모리 관리

- `_cuda_ptrs` 리스트에 CUDA 포인터 보관
- 프로그램 종료 시 `__del__` → `cudaFree` 자동 호출로 GPU 메모리 누수 방지

---

### 5.4 칼만 필터 추적기 (KalmanFilter3D)

RealSense의 노이즈 있는 3D 위치 측정값을 받아 **노이즈 제거된 위치**와 **속도**를 동시에 추정합니다.

#### 상태 모델

```
상태 벡터 (6차원): [x,  y,  z,  vx,  vy,  vz]
                    위    치        속    도

측정 벡터 (3차원): [x,  y,  z]  ← RealSense 실측값만 사용

등속도 예측 모델:
  x(t+dt)  = x(t)  + vx × dt
  y(t+dt)  = y(t)  + vy × dt
  z(t+dt)  = z(t)  + vz × dt
  vx/vy/vz = 이전 값 유지 (가속도 없다고 가정)
```

#### 핵심 행렬

```
전이 행렬 F (6×6):               측정 행렬 H (3×6):
┌ 1  0  0  dt 0  0  ┐           ┌ 1  0  0  0  0  0 ┐
│ 0  1  0  0  dt 0  │           │ 0  1  0  0  0  0 │
│ 0  0  1  0  0  dt │           └ 0  0  1  0  0  0 ┘
│ 0  0  0  1  0  0  │
│ 0  0  0  0  1  0  │
└ 0  0  0  0  0  1  ┘
※ dt는 매 프레임 실제 경과 시간으로 갱신

시스템 잡음 Q:          측정 잡음 R:        초기 오차 P:
위치항: 1×10⁻³          X/Y/Z: 1×10⁻²      위치: 1.0
속도항: 1×10⁻¹                              속도: 10.0
```

#### 2단계 동작 원리

```
[Predict — 예측]
  이전 상태(위치+속도) + 등속도 모델 → 현재 위치·속도 예측
  (RealSense 측정 없이도 관성으로 위치 추정 가능)

       ↓

[Correct — 보정]
  RealSense 실측 위치로 예측값 보정
  → 최적 추정 위치(fx, fy, fz)
  → 최적 추정 속도(vx, vy, vz) 반환
```

#### 객체 소실 처리

```
착륙 패드 미감지
    → kf_tracker.reset()
       - initialized = False
       - 상태 벡터 0으로 초기화
       - 오차 공분산 P 완전 리셋 (속도항 불확실성 10.0 복원)
    → 다음 감지 시 새로운 위치로 깨끗하게 재시작
```

---

### 5.5 RealSense 깊이 처리

#### 스트림 설정

| 채널 | 해상도 | 포맷 | FPS |
|------|--------|------|-----|
| 컬러 | 640×480 | BGR8 | 30 |
| 깊이 | 640×480 | Z16 | 30 |

#### 픽셀 정렬

```python
align = rs.align(rs.stream.color)
```
깊이 센서와 컬러 센서는 렌즈 위치가 다릅니다. `align`을 통해 두 영상의 픽셀 좌표를 **컬러 기준으로 일치**시킵니다.

#### 깊이 후처리 필터 체인

```
depth_frame
    → spatial_filter   : 인접 픽셀 공간 평균 → 깊이 홀(hole) 감소
    → temporal_filter  : 이전 프레임과 혼합  → 시간 축 노이즈 감소
    → hole_fill        : 남은 빈 픽셀 채우기
```

#### 5×5 중앙값 깊이 (`get_median_depth`)

```
중심 픽셀(cx, cy) 주변 5×5 = 25픽셀 수집
    → 0값(홀 픽셀) 제외
    → 중앙값(median) 계산
    → 유효 범위 확인: 0.15m ~ 8.0m
```

**중앙값을 사용하는 이유:** 이상치(노이즈 픽셀) 1~2개가 있어도 결과에 영향을 주지 않아 평균보다 강건합니다.

#### 유효 깊이 범위

| 파라미터 | 값 | 이유 |
|----------|----|------|
| `DEPTH_MIN_M` | 0.15m | RealSense D435i 최소 신뢰 거리 |
| `DEPTH_MAX_M` | 8.0m | 정밀 착륙 유효 고도 상한, 배경 오인식 방지 |

---

### 5.6 메인 처리 루프

#### 단계별 상세 설명

**① ~ ③ 프레임 수신 및 깊이 전처리**
```
pipeline.wait_for_frames() → 컬러+깊이 동기화된 프레임 수신
align.process()            → 깊이-컬러 픽셀 정렬
spatial/temporal/hole_fill → 깊이 품질 개선
```

**④ ~ ⑤ AI 추론 및 객체 선택**
```
trt_brain.infer(color_image)
    출력: (5, 8400) → 전치 → 8400개 후보 박스
    각 박스: [x_center, y_center, width, height, score]

score > 0.6 인 것 중 가장 높은 score 1개만 선택
    (다중 객체 시 추적 혼란 방지)
```

**⑥ 바운딩박스 좌표 복원**
```
YOLOv8은 640×640 입력으로 추론
    X축: 원본도 640px → 변환 불필요
    Y축: 원본 480px → 640px로 늘렸다가 다시 480으로 복원
         y × (480/640)
```

**⑦ ~ ⑧ 깊이 측정 및 3D 변환**
```
get_median_depth(cx, cy)
    → depth_value (미터)

rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_value)
    → raw_x : 카메라 기준 가로 오프셋 (미터, 우측 +)
    → raw_y : 카메라 기준 세로 오프셋 (미터, 하측 +)
    → raw_z : 카메라 기준 전방 거리   (미터, 전방 +)
```

**⑨ 칼만 필터 적용**
```
kf_tracker.update(raw_x, raw_y, raw_z)
    → fx, fy, fz : 노이즈 제거된 위치 (미터)
    → vx, vy, vz : 추정 속도 (m/s)
    → speed = √(vx² + vy² + vz²) : 3D 합성 속력
```

**⑩ 화면 시각화**

*아래 [7장 화면 출력 설명] 참조*

**⑪ MAVLink 송신**
```
10Hz 주기 타이머 확인
    → angle_x = atan2(fx, fz) : 가로 오프셋 각도 (라디안)
    → angle_y = atan2(fy, fz) : 세로 오프셋 각도 (라디안)
    → landing_target_send(timestamp, 0, BODY_NED, angle_x, angle_y, fz, 0, 0)
```

---

## 6. 좌표계 변환 흐름

```
[카메라 픽셀 (2D)]
  (cx, cy) — 검출 바운딩박스 중심점 (픽셀)
        │
        │  + depth_value (5×5 중앙값 깊이, 미터)
        │  + intrinsics (카메라 내부 파라미터)
        ▼
  rs2_deproject_pixel_to_point()
        │
        ▼
[카메라 3D 좌표 (미터)]
  raw_x : 우측 오프셋
  raw_y : 하측 오프셋
  raw_z : 전방 거리 (고도)
        │
        ▼
  KalmanFilter3D.update()
        │
        ▼
[칼만 보정 3D 좌표 + 속도]
  fx, fy, fz  : 노이즈 제거 위치
  vx, vy, vz  : 추정 속도 (m/s)
        │
        │  angle_x = atan2(fx, fz)
        │  angle_y = atan2(fy, fz)
        ▼
[MAVLink 각도 (라디안)]
  → Pixhawk LANDING_TARGET
  → ArduPilot PrecLand 알고리즘
  → 드론 자율 착륙 유도
```

---

## 7. 화면 출력 설명

```
┌────────────────────────────────────┐
│  Offset X: 0.12m  Y: -0.05m       │ ← 노란색: 가로/세로 오프셋
│  Alt(Z): 2.34m                     │ ← 초록색: 수직 고도
│  Vel Vx:0.03 Vy:-0.01 Vz:-0.12m/s │ ← 하늘색: 3축 속도
│  Speed: 0.12 m/s  [12.4 cm/s]     │ ← 파란색: 합성 속력
│  ┌──────────────┐                  │
│  │              │ ← 녹색 사각형: 바운딩박스
│  │      ●       │ ← 빨간 점: 객체 중심
│  │              │
│  └──────────────┘
└────────────────────────────────────┘
```

| 표시 항목 | 색상 | 단위 | 설명 |
|-----------|------|------|------|
| Offset X / Y | 노란색 | 미터 | 착륙 패드의 가로/세로 오프셋 |
| Alt(Z) | 초록색 | 미터 | 드론과 패드 간 수직 거리 |
| Vel Vx/Vy/Vz | 하늘색 | m/s | 3축 방향별 속도 |
| Speed | 파란색 | m/s / cm/s | 3D 합성 속력 |
| 녹색 사각형 | - | - | YOLO 바운딩박스 |
| 빨간 점 | - | - | 깊이 측정 중심점 |

---

## 8. 주요 설정값 튜닝 가이드

### 탐지 신뢰도 임계값

```python
best_score = 0.6  # line 265
```
- 높일수록 오탐 감소, 낮을수록 멀리서도 감지
- 권장 범위: `0.5 ~ 0.75`

### MAVLink 송신 주기

```python
MAVLINK_SEND_HZ = 10  # line 36
```
- FC 부하와 응답성 균형
- 권장 범위: `5 ~ 15 Hz`

### 유효 깊이 범위

```python
DEPTH_MIN_M = 0.15  # line 216
DEPTH_MAX_M = 8.0   # line 217
```
- 착륙 고도 범위에 맞게 조정
- 실내: `DEPTH_MAX_M = 3.0` / 실외: `DEPTH_MAX_M = 10.0`

### 칼만 필터 노이즈 행렬 튜닝

```python
# 시스템 잡음 Q (line 131-132)
q[0:3, 0:3] = np.eye(3) * 1e-3   # 위치 잡음
q[3:6, 3:6] = np.eye(3) * 1e-1   # 속도 잡음

# 측정 잡음 R (line 136)
self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
```

| 조정 방향 | 방법 | 효과 |
|-----------|------|------|
| 측정값을 더 신뢰 | R 값을 줄임 | 반응 빨라짐, 노이즈 더 탈 수 있음 |
| 모델을 더 신뢰 | R 값을 키움 | 부드러워짐, 반응 느려짐 |
| 빠른 기동 대응 | Q 속도항을 키움 | 급격한 속도 변화 추적 향상 |

### 깊이 중앙값 영역 크기

```python
get_median_depth(depth_frame, cx, cy, half=2)  # 5×5 영역
```
- `half=1` → 3×3 (9픽셀, 빠름)
- `half=2` → 5×5 (25픽셀, 기본값)
- `half=3` → 7×7 (49픽셀, 더 안정적이나 느림)

### GCS 주소 변경

```python
"udpsink host=192.168.1.30 port=5600"  # line 17
```
GCS PC의 실제 IP 주소로 변경하세요.

---

## 9. 실행 방법

### 사전 확인

```bash
# RealSense 연결 확인
rs-enumerate-devices

# TensorRT 엔진 파일 확인
ls -lh best.engine

# CUDA 라이브러리 확인
ls /usr/local/cuda/lib64/libcudart.so
```

### 실행

```bash
cd /path/to/project
python3 jetson_inference_velocity.py
```

### 종료

- 화면에서 **`q`** 키를 누르면 안전하게 종료됩니다.
- `Ctrl+C` 로도 종료 가능하나 `finally` 블록이 실행되어 하드웨어가 정상 해제됩니다.

---

## 10. 오류 대처

| 오류 메시지 | 원인 | 조치 |
|-------------|------|------|
| `⚠️ GStreamer VideoWriter 초기화 실패` | GStreamer 미설치 또는 파이프라인 오류 | GStreamer + gst-plugins-bad 설치 확인 |
| `⚠️ FC 연결 실패` | MAVLink 포트/주소 불일치 | `udp:127.0.0.1:14551` 또는 시리얼 포트 확인 |
| `best.engine` 파일 없음 | 엔진 파일 누락 | YOLOv8 → TensorRT 변환 필요 |
| 깊이값 0 지속 | RealSense 거리 밖 또는 반사면 | 조명 및 거리 확인 (0.15~8.0m) |
| 속도값 불안정 | 칼만 Q/R 튜닝 필요 | [8장 튜닝 가이드] 참조 |
| CUDA 오류 | GPU 메모리 부족 | 다른 GPU 프로세스 종료 후 재실행 |

---

*본 매뉴얼은 `jetson_inference_velocity.py v2.0` 기준으로 작성되었습니다.*
