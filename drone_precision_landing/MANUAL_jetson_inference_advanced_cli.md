# jetson_inference_advanced.py — CLI 실행 명령어 매뉴얼

**파일:** `drone_precision_landing/jetson_inference_advanced.py`  
**작성일:** 2026-06-03  
**최종 커밋:** `aa88eb6`

---

## 목차

1. [전체 CLI 인수 목록](#1-전체-cli-인수-목록)
2. [인수별 상세 설명](#2-인수별-상세-설명)
3. [단일 인수 실행 명령어](#3-단일-인수-실행-명령어)
4. [상황별 권장 조합 명령어](#4-상황별-권장-조합-명령어)
5. [인수 값 한눈에 보기](#5-인수-값-한눈에-보기)
6. [종료 방법](#6-종료-방법)

---

## 1. 전체 CLI 인수 목록

```
python3 jetson_inference_advanced.py [옵션...]

인수               선택값                        기본값
─────────────────────────────────────────────────────────────────────
--stream           on | off                      off
--model            cv | ca | ctrv | imm          cv
--tracker          single | bytetrack            single
--predict          정수 N (0=비활성)             0
--eis              on | off                      off
--eis-smoothing    정수 N (권장 5~15)            7
--motp             (플래그, 지정 시 활성)         비활성
--motp-log         on | off                      on
--traj-log         on | off                      off
--mav              연결 주소 문자열              udpin:0.0.0.0:14551
--mav-timeout      정수 SEC                      3
--aruco            on | off                      off
--aruco-dict       4X4_50|5X5_100|               4X4_50
                   6X6_250|7X7_1000
--marker-size      실수 M (미터)                 0.5
--verbose          on | off                      off
```

---

## 2. 인수별 상세 설명

### `--stream` — GStreamer UDP 스트리밍

| 값 | 동작 |
|----|------|
| `off` (기본) | 로컬 화면만 표시, GStreamer 비활성 |
| `on` | NVENC 하드웨어 인코더 자동 시도 → 실패 시 x264 폴백 → GCS PC로 UDP 전송 |

- 대상: `192.168.0.30:15600` (코드 내 고정)
- NVENC: Jetson 내장 HW 인코더, CPU 부하 없음
- x264: 소프트웨어 인코더, CPU 2~5ms 소비

---

### `--model` — 칼만 필터 모델

| 값 | 상태 차원 | 전이 방정식 | 최적 상황 | 연산 비용 |
|----|---------|------------|---------|---------|
| `cv` (기본) | 6D | 등속도 x+=vx·dt | 직선 이동, 정적 패드 | 가장 낮음 |
| `ca` | 9D | 등가속도 vx+=ax·dt | 가감속 기동 | 낮음 |
| `ctrv` | 7D | 선회율 EKF (비선형) | 드론 선회, 일반 착륙 | 중간 |
| `imm` | CV+CTRV | 확률 가중 혼합 | 복합 기동 (직선↔선회) | 높음 |

---

### `--tracker` — 추적 방식

| 값 | 동작 | 특징 |
|----|------|------|
| `single` (기본) | YOLO 출력 중 score 최고 1개만 추적 | 단순, 빠름 |
| `bytetrack` | 3단계 IoU 매칭으로 다중 트랙 관리 | ID 유지, 재진입 처리 |

---

### `--predict` — LSTM 미래 위치 예측

| 값 | 동작 |
|----|------|
| `0` (기본) | LSTM 비활성, 추가 비용 없음 |
| `5` ~ `30` | N 스텝 미래 위치 예측, 화면에 점 N개 표시 |

- 45프레임 데이터 수집 후 예측 시작
- 15프레임마다 온라인 학습 (50~200ms 블로킹 발생)
- PyTorch 미설치 시 선형 외삽으로 자동 대체

---

### `--eis` / `--eis-smoothing` — 소프트웨어 영상 안정화

| 값 | 동작 |
|----|------|
| `off` (기본) | EIS 비활성, 0ms 추가 비용 |
| `on` | Lucas-Kanade 광류 기반 진동 보정, 5~8ms 추가 |

**스무딩 윈도우 (`--eis-smoothing`):**

| 값 | 특성 | 권장 상황 |
|----|------|---------|
| `5` | 빠른 반응, 약한 안정화 | 빠른 접근 기동 |
| `7` (기본) | 균형 | 일반 착륙 |
| `10` | 강한 안정화, 느린 반응 | 진동 심한 환경 |
| `15` | 최대 안정화 | 정호버링 |

> **주의:** EIS ON 시 30fps 달성 어려움 (5~8ms 추가로 예산 초과 위험)

---

### `--motp` / `--motp-log` — 추적 정밀도 평가

| 옵션 | 동작 |
|------|------|
| `--motp` 미지정 (기본) | MOTP 평가 없음 |
| `--motp` | MOTP 실시간 계산, 화면 좌하단 표시 |
| `--motp --motp-log on` (기본) | 종료 시 `motp_log.csv` 자동 저장 |
| `--motp --motp-log off` | 화면 표시만, CSV 저장 안 함 (메모리 절약) |

---

### `--traj-log` — 궤적 로그 저장

| 값 | 동작 |
|----|------|
| `off` (기본) | 궤적 데이터 누적 없음, 비용 0 |
| `on` | 종료 시 `trajectory_log.csv` 저장 |

- 컬럼: `time_s, source, track_id, x_m, y_m, z_m, vx, vy, vz, speed_mps`
- 최대 54,000행 (30fps × 30분), 초과 시 앞 절반 자동 삭제

---

### `--mav` / `--mav-timeout` — MAVLink FC 연결

| 연결 방식 | 형식 | 특성 |
|---------|------|------|
| 기본 (수신 대기) | `udpin:0.0.0.0:14551` | 모든 인터페이스, FC 먼저 전송 |
| SITL PC (송신) | `udpout:IP:PORT` | Jetson이 먼저 연결, Heartbeat 빠름 |
| 특정 인터페이스 | `udpin:IP:14551` | 단일 인터페이스, 노이즈 감소 |
| USB-UART 직렬 | `/dev/ttyUSB0` | 가장 빠른 연결 |

**타임아웃 (`--mav-timeout`):**

| 값 | 상황 |
|----|------|
| `0` | FC 없이 즉시 진행 (영상 처리만) |
| `1` | FC 연결 확실 시 (실기체, udpout) |
| `3` (기본) | 일반 |
| `5` | FC 불확실 환경 |

---

### `--aruco` / `--aruco-dict` / `--marker-size` — ArUco 마커 탐지

| 옵션 | 동작 |
|------|------|
| `--aruco off` (기본) | ArUco 비활성 |
| `--aruco on` | 절반 해상도(320×240) 탐지 후 ×2 복원, 자세 추정 |

**소스 우선순위 (ArUco ON 시):**  
`ArUco (tvec)` > `YOLO + depth` > `YOLO (FOV 각도만)`

**ArUco 사전 (`--aruco-dict`):**

| 값 | 코드 크기 | 최대 ID 수 | 적합 거리 |
|----|---------|----------|---------|
| `4X4_50` (기본) | 4×4 | 50 | 근거리, 빠름 |
| `5X5_100` | 5×5 | 100 | 중거리 |
| `6X6_250` | 6×6 | 250 | 원거리 |
| `7X7_1000` | 7×7 | 1000 | 최대 ID |

**마커 크기 (`--marker-size`):** 실제 마커의 한 변 길이(미터). 정확도에 직접 영향.

---

### `--verbose` — 루프 내 출력 제어

| 값 | 동작 |
|----|------|
| `off` (기본) | 루프 print 없음 → 0.4~2.6ms/frame 절약 (SSH 효과 큼) |
| `on` | 🎯 탐지 결과 + 📡 MAVLink 송출 매 프레임 출력 (디버그) |

> 오류·경고 print (MAVLink 실패, LSTM 지연)는 항상 출력됨

---

## 3. 단일 인수 실행 명령어

### 3-1. 스트리밍 (`--stream`)

```bash
# 스트리밍 ON — GCS PC로 H.264 UDP 전송 (NVENC 자동, 실패 시 x264 폴백)
python3 jetson_inference_advanced.py --stream on

# 스트리밍 OFF — 로컬 모니터에만 표시, CPU 부하 감소 (기본값)
python3 jetson_inference_advanced.py --stream off
```

---

### 3-2. 칼만 모델 (`--model`)

```bash
# CV — 등속도 6D 선형 칼만, 가장 빠름, 직선 이동 패드 적합 (기본값)
python3 jetson_inference_advanced.py --model cv

# CA — 등가속도 9D 선형 칼만, 드론 가감속 기동 추적
python3 jetson_inference_advanced.py --model ca

# CTRV — 선회율 7D 비선형 EKF, 드론 선회 기동 최적, 일반 착륙 권장
python3 jetson_inference_advanced.py --model ctrv

# IMM — CV+CTRV 확률 가중 혼합, 직선↔선회 자동 전환, 복합 기동
python3 jetson_inference_advanced.py --model imm
```

---

### 3-3. 추적 방식 (`--tracker`)

```bash
# single — YOLO 출력 중 최고 score 1개만 추적, 단순하고 빠름 (기본값)
python3 jetson_inference_advanced.py --tracker single

# bytetrack — 3단계 IoU 매칭으로 다중 객체 추적, ID 유지 및 재진입 처리
python3 jetson_inference_advanced.py --tracker bytetrack
```

---

### 3-4. LSTM 미래 예측 (`--predict`)

```bash
# LSTM 비활성 — 추가 연산 없음 (기본값)
python3 jetson_inference_advanced.py --predict 0

# 5스텝 예측 — 약 0.17초(5/30fps) 미래 위치 점 5개 표시
python3 jetson_inference_advanced.py --predict 5

# 10스텝 예측 — 약 0.33초 미래 위치 점 10개 표시
python3 jetson_inference_advanced.py --predict 10

# 15스텝 예측 — 약 0.5초 미래 위치 점 15개 표시 (권장)
python3 jetson_inference_advanced.py --predict 15

# 20스텝 예측 — 약 0.67초 미래 위치 점 20개 표시
python3 jetson_inference_advanced.py --predict 20

# 30스텝 예측 — 약 1초 미래 위치 점 30개 표시 (LSTM 학습 부담 큼)
python3 jetson_inference_advanced.py --predict 30
```

---

### 3-5. EIS 영상 안정화 (`--eis`, `--eis-smoothing`)

```bash
# EIS ON — Lucas-Kanade 광류로 드론 진동 보정, 5~8ms 추가 비용
python3 jetson_inference_advanced.py --eis on

# EIS ON + 스무딩 5 — 빠른 반응, 약한 안정화 (빠른 접근 기동 적합)
python3 jetson_inference_advanced.py --eis on --eis-smoothing 5

# EIS ON + 스무딩 7 — 반응과 안정화의 균형 (기본값, 일반 착륙)
python3 jetson_inference_advanced.py --eis on --eis-smoothing 7

# EIS ON + 스무딩 10 — 강한 안정화, 느린 반응 (진동 심한 환경)
python3 jetson_inference_advanced.py --eis on --eis-smoothing 10

# EIS ON + 스무딩 15 — 최대 안정화, 정호버링 상태에서 사용
python3 jetson_inference_advanced.py --eis on --eis-smoothing 15

# EIS OFF — 안정화 없음, 0ms 추가 비용 (기본값)
python3 jetson_inference_advanced.py --eis off
```

---

### 3-6. MOTP 추적 정밀도 평가 (`--motp`, `--motp-log`)

```bash
# MOTP 활성 + CSV 저장 — 실시간 정밀도 계산 및 종료 시 motp_log.csv 저장 (기본 동작)
python3 jetson_inference_advanced.py --motp

# MOTP 활성 + CSV 저장 명시 — 위와 동일 (--motp-log on 이 기본값)
python3 jetson_inference_advanced.py --motp --motp-log on

# MOTP 활성 + CSV 저장 안 함 — 화면 표시만, 메모리 절약
python3 jetson_inference_advanced.py --motp --motp-log off
```

---

### 3-7. 궤적 로그 저장 (`--traj-log`)

```bash
# 궤적 CSV 저장 ON — 종료 시 trajectory_log.csv 에 위치·속도 이력 저장
python3 jetson_inference_advanced.py --traj-log on

# 궤적 CSV 저장 OFF — 데이터 누적 없음, 메모리 절약 (기본값)
python3 jetson_inference_advanced.py --traj-log off
```

---

### 3-8. MAVLink 연결 방식 (`--mav`, `--mav-timeout`)

```bash
# 기본 수신 대기 — 0.0.0.0:14551 에서 FC의 패킷 수신 대기 (기본값)
python3 jetson_inference_advanced.py --mav udpin:0.0.0.0:14551

# SITL PC 송신 — Jetson이 먼저 연결 요청, Heartbeat 빠르게 수신
python3 jetson_inference_advanced.py --mav udpout:192.168.0.10:14550

# 특정 인터페이스 고정 — 단일 NIC 에서만 수신, 노이즈 패킷 차단
python3 jetson_inference_advanced.py --mav udpin:192.168.0.5:14551

# USB-UART 직렬 연결 — 가장 빠르고 안정적인 FC 연결 방식
python3 jetson_inference_advanced.py --mav /dev/ttyUSB0

# 타임아웃 0초 — FC 없이 즉시 영상 처리 시작 (Heartbeat 대기 없음)
python3 jetson_inference_advanced.py --mav-timeout 0

# 타임아웃 1초 — FC 확실히 연결된 실기체 환경, 빠른 시작
python3 jetson_inference_advanced.py --mav-timeout 1

# 타임아웃 3초 — 일반적인 SITL 환경, 약간의 여유 대기 (기본값)
python3 jetson_inference_advanced.py --mav-timeout 3

# 타임아웃 5초 — FC 부팅 느리거나 연결 불확실한 환경
python3 jetson_inference_advanced.py --mav-timeout 5
```

---

### 3-9. ArUco 마커 탐지 (`--aruco`, `--aruco-dict`, `--marker-size`)

```bash
# ArUco ON — 절반 해상도(320×240) 탐지, ×2 복원, tvec 자세 추정 (기본 사전+크기)
python3 jetson_inference_advanced.py --aruco on

# 4X4_50 사전 — 4×4 코드, 최대 50 ID, 근거리 빠른 탐지 (기본값)
python3 jetson_inference_advanced.py --aruco on --aruco-dict 4X4_50

# 5X5_100 사전 — 5×5 코드, 최대 100 ID, 중거리 적합
python3 jetson_inference_advanced.py --aruco on --aruco-dict 5X5_100

# 6X6_250 사전 — 6×6 코드, 최대 250 ID, 원거리 또는 다중 마커 환경
python3 jetson_inference_advanced.py --aruco on --aruco-dict 6X6_250

# 7X7_1000 사전 — 7×7 코드, 최대 1000 ID, 대규모 마커 구분 필요 시
python3 jetson_inference_advanced.py --aruco on --aruco-dict 7X7_1000

# 마커 크기 0.3m — 30cm×30cm 마커 사용 시 (실물 크기와 반드시 일치)
python3 jetson_inference_advanced.py --aruco on --marker-size 0.3

# 마커 크기 0.5m — 50cm×50cm 마커 사용 시 (현재 사용 마커, 기본값)
python3 jetson_inference_advanced.py --aruco on --marker-size 0.5

# 마커 크기 1.0m — 1m×1m 대형 마커, 원거리 인식
python3 jetson_inference_advanced.py --aruco on --marker-size 1.0

# ArUco OFF — ArUco 탐지 비활성, YOLO+depth 방식만 사용 (기본값)
python3 jetson_inference_advanced.py --aruco off
```

---

### 3-10. 루프 출력 제어 (`--verbose`)

```bash
# verbose ON — 매 프레임 탐지 결과·MAVLink 송출 정보 터미널 출력 (디버그용)
python3 jetson_inference_advanced.py --verbose on

# verbose OFF — 루프 print 제거, 0.4~2.6ms/frame 절약, 30fps 유지 유리 (기본값)
python3 jetson_inference_advanced.py --verbose off
```

---

## 4. 상황별 권장 조합 명령어

### 4-1. 기본 실행 (옵션 없음)

```bash
# 모든 기본값 사용: stream off, cv 모델, single 추적, ArUco off, 로컬 화면만
python3 jetson_inference_advanced.py
```

---

### 4-2. 실기체 착륙 — USB-UART 직렬 연결

```bash
# 스트리밍 ON + CTRV 모델 + USB 직렬 FC 연결 (가장 빠르고 안정적인 구성)
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --mav /dev/ttyUSB0
```

---

### 4-3. 실기체 착륙 — ArUco 마커 정밀 착륙

```bash
# CTRV + ArUco 50cm 마커 탐지 → YOLO보다 정밀한 tvec 기반 자세 추정
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --aruco on \
    --marker-size 0.5 \
    --mav /dev/ttyUSB0
```

---

### 4-4. SITL 시뮬레이터 테스트

```bash
# PC SITL에 udpout으로 먼저 연결, 1초 타임아웃으로 빠른 시작
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --mav udpout:192.168.0.10:14550 \
    --mav-timeout 1
```

---

### 4-5. ByteTrack 다중 객체 추적

```bash
# ByteTrack으로 다중 착륙 패드 추적, 주 트랙(최고 score)만 MAVLink 송신
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --tracker bytetrack \
    --mav /dev/ttyUSB0
```

---

### 4-6. 드론 진동 심할 때 — EIS 활성화

```bash
# EIS smoothing 7 — 진동 보정으로 YOLO 인식률 향상, 5~8ms 추가 비용
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --aruco on \
    --marker-size 0.5 \
    --eis on \
    --eis-smoothing 7 \
    --mav udpout:192.168.0.10:14550
```

---

### 4-7. 30fps 최우선 — 최대 성능 조합

```bash
# verbose off(기본) + NVENC 자동 + udpout 빠른 연결 + 1초 타임아웃
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --aruco on \
    --marker-size 0.5 \
    --verbose off \
    --mav udpout:192.168.0.10:14550 \
    --mav-timeout 1
```

---

### 4-8. LSTM 미래 예측 활성화

```bash
# ByteTrack 주 트랙 기준 15스텝 앞 위치 예측 → 화면에 점 15개 표시
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --tracker bytetrack \
    --predict 15 \
    --mav udpout:192.168.0.10:14550
```

---

### 4-9. IMM 모델 + ByteTrack (복합 기동)

```bash
# IMM — 직선/선회 자동 전환 + ByteTrack — 다중 ID 유지
python3 jetson_inference_advanced.py \
    --stream on \
    --model imm \
    --tracker bytetrack \
    --aruco on \
    --marker-size 0.5 \
    --mav /dev/ttyUSB0
```

---

### 4-10. 연구용 풀 옵션 — 로그 전체 저장

```bash
# 모든 기능 활성 + motp_log.csv + trajectory_log.csv 동시 저장
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --tracker bytetrack \
    --aruco on \
    --marker-size 0.5 \
    --predict 15 \
    --motp \
    --motp-log on \
    --traj-log on \
    --mav udpout:192.168.0.10:14550 \
    --mav-timeout 1
```

---

### 4-11. FC 없이 영상 처리만 — 개발·테스트

```bash
# mav-timeout 0으로 FC 대기 없이 즉시 시작, verbose on으로 결과 확인
python3 jetson_inference_advanced.py \
    --stream off \
    --model cv \
    --tracker bytetrack \
    --mav-timeout 0 \
    --verbose on
```

---

### 4-12. 디버그 모드 — 모든 출력 활성

```bash
# verbose on으로 매 프레임 탐지·MAVLink 결과 출력, MOTP도 함께 표시
python3 jetson_inference_advanced.py \
    --stream off \
    --model cv \
    --verbose on \
    --motp \
    --mav-timeout 0
```

---

### 4-13. ArUco + ByteTrack + 다중 마커 환경

```bash
# 6X6_250 사전으로 최대 250개 마커 구분, IMM으로 복합 기동 대응
python3 jetson_inference_advanced.py \
    --stream on \
    --model imm \
    --tracker bytetrack \
    --aruco on \
    --aruco-dict 6X6_250 \
    --marker-size 0.5 \
    --motp \
    --mav /dev/ttyUSB0
```

---

### 4-14. 궤적 + MOTP 로그 저장 — 분석용

```bash
# 비행 후 정밀도 분석: motp_log.csv + trajectory_log.csv 동시 수집
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --tracker bytetrack \
    --motp \
    --motp-log on \
    --traj-log on \
    --mav udpout:192.168.0.10:14550
```

---

## 5. 인수 값 한눈에 보기

| 인수 | 선택 가능한 값 | 기본값 |
|------|--------------|--------|
| `--stream` | `on` `off` | `off` |
| `--model` | `cv` `ca` `ctrv` `imm` | `cv` |
| `--tracker` | `single` `bytetrack` | `single` |
| `--predict` | `0` `5` `10` `15` `20` `30` ... (정수) | `0` |
| `--eis` | `on` `off` | `off` |
| `--eis-smoothing` | `5` `7` `10` `15` ... (정수) | `7` |
| `--motp` | (플래그, 지정 시 활성) | 비활성 |
| `--motp-log` | `on` `off` | `on` |
| `--traj-log` | `on` `off` | `off` |
| `--mav` | `udpin:0.0.0.0:14551` `udpout:IP:PORT` `/dev/ttyUSBx` 등 | `udpin:0.0.0.0:14551` |
| `--mav-timeout` | `0` `1` `2` `3` `5` ... (정수) | `3` |
| `--aruco` | `on` `off` | `off` |
| `--aruco-dict` | `4X4_50` `5X5_100` `6X6_250` `7X7_1000` | `4X4_50` |
| `--marker-size` | `0.1` `0.2` `0.3` `0.5` `1.0` ... (실수, 미터) | `0.5` |
| `--verbose` | `on` `off` | `off` |

---

## 6. 종료 방법

```bash
# 방법 1 — Ctrl+C : 터미널에서 바로 사용, 가장 확실한 종료 방법
^C

# 방법 2 — q 키 : "Jetson Local View" 창을 마우스로 클릭하여 포커스 이동 후 q 입력
#           ※ 터미널에서 q를 눌러도 동작하지 않음 (cv2 창 포커스 필요)
q
```

**종료 시 자동 처리:**

```
# --motp --motp-log on 설정 시 → motp_log.csv 자동 저장
# --traj-log on 설정 시 → trajectory_log.csv 자동 저장
# RealSense 파이프라인 정상 종료 (pipeline.stop())
# GStreamer 스트림 정상 종료 (out.release())
# CUDA GPU 메모리 해제 (cudaFree)
# OpenCV 윈도우 닫기 (cv2.destroyAllWindows())
```

---

*본 매뉴얼은 `jetson_inference_advanced.py` 커밋 `aa88eb6` 기준으로 작성되었습니다.*
