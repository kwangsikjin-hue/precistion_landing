# Jetson 정밀 착륙 비전 시스템 고도화 매뉴얼

**파일명:** `jetson_inference_advanced.py`  
**버전:** v3.0 (칼만 4모델 + ByteTrack + LSTM + MOTP)  
**작성일:** 2026-06-03  
**대상 하드웨어:** NVIDIA Jetson Nano + Intel RealSense D435i + Pixhawk (ArduPilot)

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [전체 시스템 구조도](#2-전체-시스템-구조도)
3. [실행 인수 (CLI)](#3-실행-인수-cli)
4. [모듈별 상세 분석](#4-모듈별-상세-분석)
   - 4.1 TensorRT 추론 엔진
   - 4.2 칼만 필터 4모델 (CV / CA / CTRV / IMM)
   - 4.3 ByteTrack 다중 객체 추적기
   - 4.4 LSTM 미래 위치 예측기
   - 4.5 MOTP 추적 정밀도 평가기
5. [메인 루프 동작 흐름](#5-메인-루프-동작-흐름)
   - 5.1 공통 전처리 단계
   - 5.2 모드 A: single (단일 객체)
   - 5.3 모드 B: bytetrack (다중 객체)
6. [칼만 모델 비교 및 선택 가이드](#6-칼만-모델-비교-및-선택-가이드)
7. [화면 출력 설명](#7-화면-출력-설명)
8. [주요 설정값 튜닝 가이드](#8-주요-설정값-튜닝-가이드)
9. [실행 방법 및 예시](#9-실행-방법-및-예시)
10. [오류 대처](#10-오류-대처)

---

## 1. 시스템 개요

본 소프트웨어는 **드론 자율 정밀 착륙(Precision Landing)** 을 위한 Jetson Nano 엣지 비전 시스템입니다.

### 핵심 기능 5가지

| 번호 | 기능 | 설명 |
|------|------|------|
| ① | **AI 객체 탐지** | YOLOv8 + TensorRT GPU 가속 실시간 추론 |
| ② | **3D 위치 추정** | RealSense D435i 깊이 카메라 역투영 |
| ③ | **칼만 필터 추적** | CV / CA / CTRV / IMM 4가지 모델 선택 가능 |
| ④ | **다중 객체 추적** | ByteTrack으로 여러 객체에 영구 ID 부여 |
| ⑤ | **LSTM 미래 예측** | 과거 궤적 학습 후 N 프레임 앞 위치 예측 |

추가로 **MAVLink LANDING_TARGET** 송신으로 Pixhawk 자동 착륙 유도,  
**GStreamer UDP** 스트리밍으로 GCS(지상국) 실시간 영상 전송을 수행합니다.

---

## 2. 전체 시스템 구조도

```
┌──────────────────────────────────────────────────────────────────┐
│                        Jetson Nano                               │
│                                                                  │
│  ┌──────────────────┐   ┌─────────────────────────────────────┐  │
│  │  RealSense D435i │   │          TensorRT GPU               │  │
│  │  컬러: 640×480   │──►│  YOLOv8 best.engine                 │  │
│  │  깊이: 640×480   │   │  8400개 바운딩박스 출력              │  │
│  │  30fps           │   └──────────────┬──────────────────────┘  │
│  └──────────────────┘                  │                         │
│           │                            │ all_dets                │
│           │ depth_frame                ▼                         │
│   ┌───────┴───────────────────────────────────────────────┐     │
│   │              추적 모드 선택 (--tracker)                │     │
│   │  ┌─────────────────┐    ┌────────────────────────────┐│     │
│   │  │ single 모드      │    │ bytetrack 모드             ││     │
│   │  │ 최고신뢰도 1개   │    │ ByteTracker (3단계 매칭)   ││     │
│   │  │ single_kf 사용  │    │ 트랙별 track_3d_kfs 사용   ││     │
│   │  └────────┬────────┘    └──────────────┬─────────────┘│     │
│   └───────────┼──────────────────────────┬─┘              │     │
│               │                          │                 │     │
│               ▼                          ▼                 │     │
│   ┌─────────────────────────────────────────────────────┐ │     │
│   │    칼만 필터 모델 (--model)                          │ │     │
│   │    CV(6D) / CA(9D) / CTRV(7D EKF) / IMM(혼합)      │ │     │
│   │    출력: (fx,fy,fz, vx,vy,vz, innovation_norm)      │ │     │
│   └──────────────────────────┬──────────────────────────┘ │     │
│                               │                             │     │
│          ┌────────────────────┼────────────────────┐        │     │
│          ▼                    ▼                    ▼        │     │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────┐  │     │
│  │ LSTM 예측기  │  │ MOTP 평가기     │  │ 화면 시각화  │  │     │
│  │ N스텝 미래   │  │ innovation 누적 │  │ 오버레이     │  │     │
│  │ 궤적 점 표시 │  │ 평균 정밀도     │  │ 텍스트 4줄   │  │     │
│  └──────────────┘  └─────────────────┘  └──────────────┘  │     │
│                                                              │     │
│          ┌─────────────────────────────────────┐            │     │
│          │ MAVLink LANDING_TARGET (10Hz)        │            │     │
│          │ → Pixhawk FC 자동 착륙 유도           │            │     │
│          └─────────────────────────────────────┘            │     │
│          ┌─────────────────────────────────────┐            │     │
│          │ GStreamer H.264 UDP (30fps)          │            │     │
│          │ → GCS PC 192.168.1.30:5600          │            │     │
│          └─────────────────────────────────────┘            │     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 실행 인수 (CLI)

### 기본 사용법

```bash
python3 jetson_inference_advanced.py [옵션들]
```

### 전체 옵션 목록

| 옵션 | 값 | 기본값 | 설명 |
|------|----|--------|------|
| `--model` | `cv` `ca` `ctrv` `imm` | `cv` | 칼만 필터 모델 선택 |
| `--tracker` | `single` `bytetrack` | `single` | 추적 방식 선택 |
| `--predict` | 정수 N | `0` | LSTM 미래 예측 스텝 수 (0=비활성) |
| `--motp` | 플래그 | 비활성 | MOTP 추적 정밀도 평가 활성화 |

### 실행 예시

```bash
# 기본 실행 (등속도 모델, 단일 추적)
python3 jetson_inference_advanced.py

# CTRV 모델 + ByteTrack 다중 추적 + LSTM 15스텝 예측 + MOTP 평가
python3 jetson_inference_advanced.py --model ctrv --tracker bytetrack --predict 15 --motp

# IMM 복합 모델 + ByteTrack
python3 jetson_inference_advanced.py --model imm --tracker bytetrack

# 등가속도 모델 + 단일 추적 + MOTP 평가
python3 jetson_inference_advanced.py --model ca --motp

# 도움말 확인
python3 jetson_inference_advanced.py --help
```

---

## 4. 모듈별 상세 분석

### 4.1 TensorRT 추론 엔진 (`JetsonTRTEngine`)

YOLOv8 모델을 Jetson GPU에서 최고 속도로 실행하는 추론 엔진입니다.

#### 초기화 순서

```
best.engine 파일 로드
    → TensorRT Runtime 역직렬화
    → 실행 컨텍스트 생성
    → 입력/출력 버퍼에 cudaMalloc (GPU 메모리 할당)
    → 포인터 _ptrs에 저장 (소멸자에서 cudaFree)
```

#### 버퍼 형상

| 버퍼 | 형상 | 설명 |
|------|------|------|
| 입력 | `(1, 3, 640, 640)` | 배치1, RGB 채널, 640×640 |
| 출력 | `(1, 5, 8400)` | 8400 앵커 × [x, y, w, h, score] |

#### 추론 흐름 (`infer`)

```
컬러 이미지 (640×480 BGR)
    ↓ cv2.resize → 640×640
    ↓ transpose(2,0,1) → CHW 변환
    ↓ /255.0 → 0.0~1.0 정규화
    ↓ cudaMemcpy (Host→GPU)
    ↓ execute_v2 (GPU 추론)
    ↓ cudaMemcpy (GPU→Host)
    → (1, 5, 8400) 배열 반환
```

#### 메모리 관리

- `_ptrs` 리스트에 CUDA 포인터 보관
- `__del__` 소멸자에서 `cudaFree` 자동 호출 → 메모리 누수 방지

---

### 4.2 칼만 필터 4모델

모든 모델은 동일한 인터페이스를 따릅니다.

```
update(x, y, z) → (fx, fy, fz, vx, vy, vz, innovation_norm)
```

| 반환값 | 의미 |
|--------|------|
| `fx, fy, fz` | 칼만 보정 3D 위치 (m) |
| `vx, vy, vz` | 추정 3D 속도 (m/s) |
| `innovation_norm` | 예측↔측정 거리 오차 (m), MOTP에 사용 |

---

#### 모델 A: CV — Constant Velocity (등속도) 6D 선형

```
상태 벡터: [x, y, z, vx, vy, vz]
           위    치      속    도

전이 방정식:
  x(t+dt) = x(t) + vx × dt
  vx(t+dt) = vx(t)          ← 속도 불변 가정

구현: cv2.KalmanFilter (선형, 빠름)
```

**적합 상황:** 착륙 패드가 정지 또는 직선으로 천천히 이동할 때

---

#### 모델 B: CA — Constant Acceleration (등가속도) 9D 선형

```
상태 벡터: [x, y, z, vx, vy, vz, ax, ay, az]
           위    치      속    도      가속도

전이 방정식:
  x(t+dt) = x(t) + vx×dt + 0.5×ax×dt²
  vx(t+dt) = vx(t) + ax×dt
  ax(t+dt) = ax(t)          ← 가속도 불변 가정

구현: cv2.KalmanFilter (선형, dt마다 F 행렬 갱신)
```

**적합 상황:** 드론이 가속/감속 기동을 할 때

---

#### 모델 C: CTRV — Constant Turn Rate & Velocity (선회율) 7D EKF

```
상태 벡터: [x, y, z, v, yaw, yaw_rate, vz]
           위 치  수평속도 방위각  선회율  수직속도

전이 방정식 (비선형):
  ┌ yaw_rate ≈ 0 (직선):
  │   x += v×cos(yaw)×dt
  │   y += v×sin(yaw)×dt
  │
  └ yaw_rate ≠ 0 (선회):
      x += (v/ω) × (sin(yaw+ω×dt) - sin(yaw))
      y += (v/ω) × (-cos(yaw+ω×dt) + cos(yaw))

구현: Extended Kalman Filter (EKF)
      - 비선형 → 야코비안(Jacobian) 행렬 직접 계산
      - 직선/선회 자동 분기 (임계값 ε = 1×10⁻⁴)
```

**야코비안 계산 (7×7 행렬):**

직선 분기 주요 항:
```
F[0,3] = cos(yaw)×dt      ← ∂x/∂v
F[0,4] = -v×sin(yaw)×dt   ← ∂x/∂yaw
F[1,3] = sin(yaw)×dt
F[1,4] = v×cos(yaw)×dt
```

선회 분기 주요 항:
```
F[0,3] = (sin(yaw+ω×dt) - sin(yaw)) / ω    ← ∂x/∂v
F[0,4] = (v/ω) × (cos(yaw+ω×dt) - cos(yaw)) ← ∂x/∂yaw
F[0,5] = v × (dt×cos(yaw+ω×dt)/ω - (sin(yaw+ω×dt)-sin(yaw))/ω²)
```

**적합 상황:** 드론이 선회하며 이동하거나, 패드가 기동하는 상황

---

#### 모델 D: IMM — Interacting Multiple Models (복합 모델)

```
구성: CV + CTRV 두 필터를 확률 μ로 가중 혼합

IMM 4단계:
  1. 혼합 (Mixing)
     c_j = Σ_i PI[i,j] × μ[i]   (정규화 상수)
  
  2. 모드별 독립 필터링
     CV.update(x,y,z)   → result_CV
     CTRV.update(x,y,z) → result_CTRV
  
  3. 확률 갱신 (우도 기반)
     L[j] = exp(-0.5 × (innov[j]/σ)²)    (가우시안 우도)
     μ_new = L × c / Σ(L × c)
  
  4. 추정값 융합
     fx = μ[0]×result_CV[0] + μ[1]×result_CTRV[0]
     (vx, vy, vz도 동일하게 가중 평균)

전환 행렬 PI:
  [[0.95, 0.05],    ← CV 유지 95%, CTRV 전환 5%
   [0.05, 0.95]]    ← CTRV 유지 95%, CV 전환 5%
```

**동작 예:**
- 직선 이동 → μ_CV ↑, μ_CTRV ↓ (CV 우세)
- 선회 기동 → μ_CV ↓, μ_CTRV ↑ (CTRV 우세)
- 화면에 `IMM CV:0.72 CTRV:0.28` 형태로 현재 확률 표시

**적합 상황:** 직선/선회를 반복하는 복잡한 기동 전반

---

### 4.3 ByteTrack 다중 객체 추적기

ByteTrack은 YOLO 검출 박스에 **영구 ID**를 부여하고 프레임 간 동일 객체를 연결합니다.

#### 관련 클래스 3개

```
KalmanBoxFilter   ← 2D 바운딩박스 칼만 (ByteTrack 전용)
STrack            ← 단일 트랙 객체
ByteTracker       ← 전체 추적 관리자
```

---

#### `KalmanBoxFilter` — 2D 바운딩박스 전용 칼만

```
상태: [cx, cy, a, h, vcx, vcy, va, vh]
       중심좌표  종횡비 높이  속도항들
측정: [cx, cy, a, h]

※ 3D 칼만 모델과 별개로 ByteTrack 내부에서만 사용
   픽셀 공간에서 박스 위치를 추적하여 ID 매칭에 사용
```

---

#### `STrack` — 단일 트랙 생명 주기

```
생성 (new)
    ↓ 2프레임 이상 연속 매칭
확인 (is_confirmed = True) → 화면에 표시
    ↓ 매칭 실패
소실 (lost) → max_lost=30프레임 동안 예측으로 유지
    ↓ max_lost 초과
제거 (삭제)

재진입: lost 상태에서 다시 매칭되면 → tracked 복귀
```

---

#### `ByteTracker.update()` — 3단계 매칭 알고리즘

```
입력 검출박스 분류:
  high: score ≥ 0.6  (고신뢰도)
  low:  0.3 ≤ score < 0.6  (저신뢰도)

┌─ 1차 매칭 ──────────────────────────────────────────┐
│ 활성 트랙 ↔ 고신뢰도 검출                            │
│ 비용 행렬: cost[i,j] = 1 - IoU(track_i, det_j)      │
│ 최적화: 헝가리안 알고리즘 (scipy) 또는 그리디 폴백    │
│ 임계값: IoU ≥ 0.8 (match_thresh)                    │
│ → 매칭 성공: track.update(det)                      │
│ → 미매칭 트랙: 2차로 넘김                            │
│ → 미매칭 검출: 3차로 넘김                            │
└─────────────────────────────────────────────────────┘
        ↓
┌─ 2차 매칭 ──────────────────────────────────────────┐
│ 미매칭 활성 트랙 ↔ 저신뢰도 검출                     │
│ 임계값: IoU ≥ 0.5 (완화됨)                          │
│ → 매칭 성공: track.update(det)                      │
│ → 미매칭 트랙: 소실(lost) 처리                       │
└─────────────────────────────────────────────────────┘
        ↓
┌─ 3차 매칭 ──────────────────────────────────────────┐
│ 소실 트랙 ↔ 남은 고신뢰도 검출 (재진입 감지)          │
│ → 매칭 성공: tracked 복귀                            │
│ → 미매칭 검출: 신규 트랙 생성 (새 ID 부여)           │
└─────────────────────────────────────────────────────┘

최종 반환: is_confirmed(tracklet_len ≥ 2) 트랙만 반환
```

#### ByteTrack의 핵심 장점

- **저신뢰도 활용**: 가림(occlusion) 시 약하게 보이는 박스도 2차 매칭에서 트랙 유지
- **재진입 처리**: 화면 밖으로 나갔다 돌아온 객체를 같은 ID로 재연결
- **고신뢰도 우선**: 1차에서 확실한 매칭 먼저 처리 → 주 트랙(MAVLink 송신) 안정성

#### 트랙별 3D 칼만 필터

ByteTrack이 ID를 부여하면, 각 ID마다 독립적인 3D 칼만 필터가 생성됩니다.

```python
track_3d_kfs = {}   # {track_id: KFModel 인스턴스}

# 새 트랙 → 새 3D KF 생성
if tid not in track_3d_kfs:
    track_3d_kfs[tid] = create_tracker(args.model)

# 소멸 트랙 → 3D KF 삭제
for tid not in active_ids:
    del track_3d_kfs[tid]
```

---

### 4.4 LSTM 기반 미래 위치 예측기 (`LSTMPredictor`)

과거 궤적 데이터로 LSTM을 온라인 학습하여 미래 N 프레임 위치를 예측합니다.

#### 네트워크 구조 (`_LSTMNet`)

```
입력:  (배치, SEQ_LEN=30, 3)   ← 과거 30프레임 (x,y,z) 시퀀스
         ↓
LSTM:  input_size=3 → hidden=64 → num_layers=2 → dropout=0.1
         ↓
Linear: 64 → predict_steps × 3
         ↓
출력:  (배치, predict_steps, 3)   ← 미래 N프레임 (x,y,z) 예측
```

#### 온라인 학습 절차

```
① 칼만 보정 위치(fx,fy,fz)를 buf(최대 120개)에 누적
   - 첫 위치를 ref로 저장 후 정규화 (ref 기준 상대좌표)

② 조건 충족 시 학습 실행:
   buf 크기 ≥ SEQ_LEN(30) + predict_steps AND
   매 TRAIN_EVERY(15)프레임마다

③ 슬라이딩 윈도우 학습:
   [0:30] → [30:30+N]  (입력→타겟 쌍 생성)
   [1:31] → [31:31+N]
   ...
   Adam 옵티마이저, MSE 손실, 5 에폭

④ 예측:
   최근 30프레임 → LSTM → 미래 N프레임 (x,y,z)
   화면에 반투명 점으로 표시 (가까운 예측일수록 크고 밝음)
```

#### PyTorch 미설치 시 폴백

```
최근 5프레임의 선형 추세로 외삽 (Linear Extrapolation)
  vel = (tail[-1] - tail[0]) / 4
  future[k] = tail[-1] + vel × k
```

#### LSTM 적용 범위

| 모드 | LSTM 적용 대상 |
|------|--------------|
| single | 유일한 추적 객체 |
| bytetrack | **주 트랙(highest score) 1개에만** 적용 |

---

### 4.5 MOTP 추적 정밀도 평가기 (`MOTPEvaluator`)

```
MOTP = Σ d_i / Σ c_i

d_i : 프레임 i의 innovation_norm (칼만 예측↔RealSense 측정 거리, m)
c_i : 프레임 i의 매칭 수 (단일=1, 다중=활성 트랙 수)

값이 작을수록 → 칼만 필터가 다음 위치를 정확히 예측 → 추적 정밀도 높음
```

> **참고:** 정식 MOTP는 Ground-Truth 레이블과 비교하지만, 여기서는 칼만 Innovation을 추적 정밀도 근사값으로 사용합니다.

#### 출력 및 저장

- 화면 좌하단에 `MOTP:2.34cm (N=150)` 형태로 실시간 표시
- 터미널에도 함께 출력
- 종료 시 `motp_log.csv` 자동 저장 (프레임별 innovation_cm)

---

## 5. 메인 루프 동작 흐름

### 5.1 공통 전처리 단계 (매 프레임)

```
① RealSense 프레임 수신
   pipeline.wait_for_frames() → 컬러+깊이 동기화

② 깊이-컬러 픽셀 정렬
   align.process() → 깊이 픽셀과 컬러 픽셀 좌표 일치

③ 깊이 후처리 필터 체인
   spatial_filter  → 인접 픽셀 공간 평균 (홀 감소)
   temporal_filter → 이전 프레임 혼합 (시간 노이즈 감소)
   hole_fill       → 남은 빈 픽셀 채우기

④ TensorRT 추론
   color_image → trt_brain.infer() → (1,5,8400) → reshape → 8400개 예측

⑤ 검출 박스 수집 (score ≥ 0.3)
   640×640 좌표 → 640×480 좌표로 변환
   all_dets = [[x1,y1,x2,y2,score], ...]
```

---

### 5.2 모드 A: single (단일 객체 추적)

```
⑥-A 최고 신뢰도 선택
     best = max(all_dets, key=score)
     조건: score ≥ 0.6

⑦-A 5×5 중앙값 깊이 측정
     get_median_depth(depth_frame, cx, cy)
     유효 범위: 0.15m ~ 8.0m

⑧-A 2D → 3D 역투영
     rs2_deproject_pixel_to_point(intrinsics, [cx,cy], depth)
     → raw_x, raw_y, raw_z (미터)

⑨-A 단일 3D 칼만 업데이트
     single_kf.update(raw_x, raw_y, raw_z)
     → fx,fy,fz (보정 위치), vx,vy,vz (속도), innov

⑩-A MOTP 갱신 (--motp 활성 시)
     motp_eval.update(innov)

⑪-A LSTM 업데이트 및 예측 (--predict N 활성 시)
     lstm_pred.add(fx,fy,fz) → 버퍼에 추가
     futures = lstm_pred.predict() → N개 미래 위치

⑫-A 화면 시각화
     녹색 사각형: 바운딩박스
     빨간 점: 검출 중심
     텍스트 4줄: 위치/고도/속도/속력
     LSTM 점: 미래 궤적 (보라색~밝은보라)
     MOTP: 좌하단

⑬-A MAVLink 10Hz 송신
     angle_x = atan2(fx, fz)
     angle_y = atan2(fy, fz)
     → LANDING_TARGET 메시지

⑭-A 객체 소실 시
     single_kf.reset() + lstm_pred.reset()
```

---

### 5.3 모드 B: bytetrack (다중 객체 추적)

```
⑥-B ByteTracker 3단계 매칭
     byte_tracker.update(all_dets)
     → active: 확인된 활성 트랙 목록

⑦-B 각 활성 트랙 처리 (루프)
     ├─ track.center → cx, cy (칼만 예측 중심점)
     ├─ get_median_depth(cx, cy) → 깊이 측정
     ├─ rs2_deproject → raw_x, raw_y, raw_z
     ├─ track_3d_kfs[tid].update() → fx,fy,fz,vx,vy,vz,innov
     ├─ MOTP 갱신
     ├─ 화면 시각화 (ID별 고유 색상)
     │    색상 박스 + "ID{n} {score:.2f}" + 위치/속도 텍스트
     └─ primary 후보 갱신 (최고 score 트랙 선택)

⑧-B 주 트랙 (primary) 처리
     ├─ LSTM: 주 트랙에만 lstm_pred.add() / predict()
     ├─ LSTM 점: 화면에 미래 궤적 표시
     └─ MAVLink: 주 트랙 좌표로 10Hz 송신

⑨-B 소멸 트랙 정리
     active_ids = {t.track_id for t in active}
     active_ids에 없는 track_3d_kfs 항목 삭제

⑩-B 공통 오버레이
     좌상단: "ByteTrack | CTRV | 활성:3"
     좌하단: MOTP 값
```

---

## 6. 칼만 모델 비교 및 선택 가이드

### 모델 특성 비교

| 항목 | CV | CA | CTRV | IMM |
|------|----|----|------|-----|
| 상태 차원 | 6D | 9D | 7D | 13D (CV6+CTRV7) |
| 수학 | 선형 KF | 선형 KF | 비선형 EKF | 확률 혼합 |
| 연산 비용 | 낮음 | 중간 | 중간 | 높음 |
| 직선 이동 | ★★★ | ★★★ | ★★★ | ★★★ |
| 가감속 | ★★☆ | ★★★ | ★★☆ | ★★★ |
| 선회 기동 | ★☆☆ | ★★☆ | ★★★ | ★★★ |
| 복합 기동 | ★☆☆ | ★★☆ | ★★☆ | ★★★ |

### 상황별 권장 모델

| 사용 상황 | 권장 모델 |
|-----------|-----------|
| 정지한 착륙 패드 | `cv` |
| 서서히 이동하는 패드 | `cv` 또는 `ca` |
| 드론이 직선으로 접근 | `cv` |
| 드론이 선회하며 접근 | `ctrv` |
| 패드가 불규칙 이동 | `imm` |
| 다양한 기동 연구 비교 | `imm` |

### CTRV 화면 표시 해석

```
Model: CTRV[직선]          ← yaw_rate < 1e-4 rad/s → 직선 분기
Model: CTRV[선회(15.3°/s)] ← yaw_rate ≥ 1e-4 → 선회 분기, 초당 선회율
```

### IMM 화면 표시 해석

```
Model: IMM CV:0.82 CTRV:0.18  ← CV가 지배 (직선 이동 중)
Model: IMM CV:0.23 CTRV:0.77  ← CTRV가 지배 (선회 이동 중)
```

---

## 7. 화면 출력 설명

### single 모드 화면

```
┌────────────────────────────────────────────────────┐
│ Model: CTRV[직선]                      (좌상단)     │
│                                                    │
│  ┌──────────────────┐                              │
│  │                  │ ← 녹색 사각형: 바운딩박스    │
│  │  Offset X:0.12m  Y:-0.05m   (노란색)            │
│  │  Alt(Z): 2.34m              (초록색)            │
│  │  Vel Vx:0.03 Vy:-0.01 Vz:-0.12 m/s (하늘색)   │
│  │  Speed:0.13m/s [13cm/s]     (파란색)            │
│  │        ●                    (빨간 점: 중심)     │
│  └──────────────────┘                              │
│    ● ● ●  (보라색 점: LSTM 미래 궤적)              │
│                                                    │
│ MOTP:2.34cm (N=150)                    (좌하단)    │
└────────────────────────────────────────────────────┘
```

### bytetrack 모드 화면

```
┌────────────────────────────────────────────────────┐
│ ByteTrack | CTRV | 활성:3              (좌상단)     │
│                                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ (파란색) │    │ (주황색) │    │ (초록색) │      │
│  │ID1  0.91│    │ID2  0.76│    │ID3  0.62│      │
│  │ Offset..│    │ Offset..│    │ Offset..│      │
│  │   ●     │    │   ●     │    │   ●     │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│  ↑ 주 트랙 (최고 신뢰도)                           │
│  LSTM 미래 궤적 점 (주 트랙 기준)                  │
│                                                    │
│ MOTP:1.87cm (N=320)                    (좌하단)    │
└────────────────────────────────────────────────────┘
```

### 화면 요소 색상 정의

| 색상 | 의미 |
|------|------|
| 녹색 사각형 | single 모드 바운딩박스 |
| 트랙별 고유색 | bytetrack 모드 각 트랙 |
| 빨간 점 | 검출 중심점 |
| 노란색 텍스트 | 가로/세로 오프셋 (m) |
| 초록색 텍스트 | 수직 고도 Z (m) |
| 하늘색 텍스트 | 3축 속도 (m/s) |
| 파란색 텍스트 | 합성 속력 (m/s, cm/s) |
| 밝은 보라색 점 | LSTM 가까운 미래 예측 |
| 어두운 보라색 점 | LSTM 먼 미래 예측 |
| 연초록 텍스트 | MOTP 값 (좌하단) |

---

## 8. 주요 설정값 튜닝 가이드

### YOLO 신뢰도 임계값

```python
# 검출 박스 수집 최소값 (all_dets 구성)
score < 0.3   → 버림 (라인 768)

# single 모드 최종 사용 임계값
score >= 0.6  → 사용 (라인 781)

# bytetrack 고신뢰도/저신뢰도 구분
track_thresh  = 0.6  (1차 매칭)
second_thresh = 0.3  (2차 매칭)
```

### 칼만 필터 노이즈 행렬

| 행렬 | 기본값 | 조정 방향 |
|------|--------|-----------|
| Q (위치항) | 1×10⁻³ | 크게 → 측정값 추종 빨라짐 |
| Q (속도항) | 1×10⁻¹ | 크게 → 속도 변화 빠르게 반응 |
| R (측정잡음) | 1×10⁻² | 크게 → 노이즈 제거 강화, 반응 느림 |

### ByteTrack 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `track_thresh` | 0.6 | 고신뢰도 1차 매칭 임계값 |
| `second_thresh` | 0.3 | 저신뢰도 2차 매칭 하한 |
| `match_thresh` | 0.8 | IoU 매칭 임계값 |
| `max_lost` | 30 | 소실 트랙 유지 최대 프레임 수 |

### LSTM 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `SEQ_LEN` | 30 | 입력 시퀀스 길이 (프레임) |
| `TRAIN_EVERY` | 15 | 학습 주기 (프레임) |
| `EPOCHS` | 5 | 1회 학습 에폭 수 |
| `buf maxlen` | 120 | 궤적 히스토리 최대 길이 |

### MAVLink 송신 주기

```python
MAVLINK_SEND_HZ = 10   # 10Hz (100ms 간격)
# 범위: 5~15Hz 권장
```

### 유효 깊이 범위

```python
DEPTH_MIN_M = 0.15   # 최소 신뢰 거리 (D435i 사양)
DEPTH_MAX_M = 8.0    # 정밀 착륙 유효 고도 상한
# 실내 작업: 3.0m / 야외 고고도: 10.0m
```

---

## 9. 실행 방법 및 예시

### 사전 확인

```bash
# RealSense 연결 확인
rs-enumerate-devices

# TensorRT 엔진 확인
ls -lh best.engine

# CUDA 라이브러리 확인
ls /usr/local/cuda/lib64/libcudart.so

# PyTorch 설치 확인 (LSTM 사용 시)
python3 -c "import torch; print(torch.__version__)"

# scipy 설치 확인 (ByteTrack 최적 매칭)
python3 -c "from scipy.optimize import linear_sum_assignment; print('OK')"
```

### 추천 실행 조합

```bash
# [초보] 기본 실행 — 단순하고 빠름
python3 jetson_inference_advanced.py

# [연구] 선회 추적 성능 비교
python3 jetson_inference_advanced.py --model ctrv --motp
python3 jetson_inference_advanced.py --model cv   --motp
# → motp_log.csv 비교로 CTRV vs CV 정밀도 측정

# [다중 추적] ByteTrack + 복합 모델
python3 jetson_inference_advanced.py --model imm --tracker bytetrack

# [풀 옵션] 모든 기능 활성화
python3 jetson_inference_advanced.py \
    --model ctrv \
    --tracker bytetrack \
    --predict 20 \
    --motp
```

### 종료

- 화면에서 **`q`** 키 → 안전 종료 (motp_log.csv 저장)
- `Ctrl+C` → finally 블록 실행 후 종료

---

## 10. 오류 대처

| 오류 메시지 / 증상 | 원인 | 조치 |
|-------------------|------|------|
| `⚠️ GStreamer 초기화 실패` | GStreamer 미설치 또는 IP 오류 | GStreamer 설치 확인, host IP 변경 |
| `⚠️ FC 연결 실패` | MAVLink 포트 불일치 | UDP 포트 또는 시리얼 포트 확인 |
| `best.engine` 파일 없음 | 모델 파일 누락 | YOLOv8 → TensorRT 변환 실행 |
| 속도값 매우 불안정 | Q/R 튜닝 필요 | R 값 증가 (노이즈 제거 강화) |
| LSTM 예측 비활성화 상태 지속 | 데이터 30프레임 미달 | 객체가 30프레임 이상 연속 감지될 때 활성화 |
| ByteTrack ID 잦은 변경 | match_thresh 낮음 | `match_thresh` 0.7→0.8로 상향 |
| `ℹ️ LSTM → 선형 외삽 대체` | PyTorch 미설치 | `pip install torch torchvision` |
| MOTP 값 매우 큼 (>10cm) | 칼만 발산 또는 RealSense 불안정 | reset() 호출, 조명·거리 확인 |
| CUDA 오류 | GPU 메모리 부족 | 다른 GPU 프로세스 종료 후 재실행 |

---

*본 매뉴얼은 `jetson_inference_advanced.py v3.0` 기준으로 작성되었습니다.*
