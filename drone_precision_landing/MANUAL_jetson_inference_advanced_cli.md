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
python3 jetson_inference_advanced.py --stream on
python3 jetson_inference_advanced.py --stream off
```

### 3-2. 칼만 모델 (`--model`)
```bash
python3 jetson_inference_advanced.py --model cv
python3 jetson_inference_advanced.py --model ca
python3 jetson_inference_advanced.py --model ctrv
python3 jetson_inference_advanced.py --model imm
```

### 3-3. 추적 방식 (`--tracker`)
```bash
python3 jetson_inference_advanced.py --tracker single
python3 jetson_inference_advanced.py --tracker bytetrack
```

### 3-4. LSTM 예측 (`--predict`)
```bash
python3 jetson_inference_advanced.py --predict 0    # 비활성 (기본)
python3 jetson_inference_advanced.py --predict 5
python3 jetson_inference_advanced.py --predict 10
python3 jetson_inference_advanced.py --predict 15
python3 jetson_inference_advanced.py --predict 20
python3 jetson_inference_advanced.py --predict 30
```

### 3-5. EIS 안정화 (`--eis`, `--eis-smoothing`)
```bash
python3 jetson_inference_advanced.py --eis on
python3 jetson_inference_advanced.py --eis on --eis-smoothing 5
python3 jetson_inference_advanced.py --eis on --eis-smoothing 7
python3 jetson_inference_advanced.py --eis on --eis-smoothing 10
python3 jetson_inference_advanced.py --eis on --eis-smoothing 15
python3 jetson_inference_advanced.py --eis off
```

### 3-6. MOTP 평가 (`--motp`, `--motp-log`)
```bash
python3 jetson_inference_advanced.py --motp
python3 jetson_inference_advanced.py --motp --motp-log on
python3 jetson_inference_advanced.py --motp --motp-log off
```

### 3-7. 궤적 로그 (`--traj-log`)
```bash
python3 jetson_inference_advanced.py --traj-log on
python3 jetson_inference_advanced.py --traj-log off
```

### 3-8. MAVLink 연결 (`--mav`, `--mav-timeout`)
```bash
python3 jetson_inference_advanced.py --mav udpin:0.0.0.0:14551
python3 jetson_inference_advanced.py --mav udpout:192.168.0.10:14550
python3 jetson_inference_advanced.py --mav udpin:192.168.0.5:14551
python3 jetson_inference_advanced.py --mav /dev/ttyUSB0
python3 jetson_inference_advanced.py --mav-timeout 0
python3 jetson_inference_advanced.py --mav-timeout 1
python3 jetson_inference_advanced.py --mav-timeout 3
python3 jetson_inference_advanced.py --mav-timeout 5
```

### 3-9. ArUco 탐지 (`--aruco`, `--aruco-dict`, `--marker-size`)
```bash
python3 jetson_inference_advanced.py --aruco on
python3 jetson_inference_advanced.py --aruco on --aruco-dict 4X4_50
python3 jetson_inference_advanced.py --aruco on --aruco-dict 5X5_100
python3 jetson_inference_advanced.py --aruco on --aruco-dict 6X6_250
python3 jetson_inference_advanced.py --aruco on --aruco-dict 7X7_1000
python3 jetson_inference_advanced.py --aruco on --marker-size 0.3
python3 jetson_inference_advanced.py --aruco on --marker-size 0.5
python3 jetson_inference_advanced.py --aruco on --marker-size 1.0
python3 jetson_inference_advanced.py --aruco off
```

### 3-10. 출력 제어 (`--verbose`)
```bash
python3 jetson_inference_advanced.py --verbose on
python3 jetson_inference_advanced.py --verbose off
```

---

## 4. 상황별 권장 조합 명령어

### 4-1. 기본 실행 (옵션 없음)
```bash
python3 jetson_inference_advanced.py
```

---

### 4-2. 실기체 착륙 — USB-UART 직렬 연결
```bash
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --mav /dev/ttyUSB0
```

---

### 4-3. 실기체 착륙 — ArUco 마커 정밀 착륙
```bash
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
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --mav udpout:192.168.0.10:14550 \
    --mav-timeout 1
```

---

### 4-5. ByteTrack 다중 객체 추적
```bash
python3 jetson_inference_advanced.py \
    --stream on \
    --model ctrv \
    --tracker bytetrack \
    --mav /dev/ttyUSB0
```

---

### 4-6. 드론 진동 심할 때 — EIS 활성화
```bash
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
# 방법 1 — Ctrl+C (터미널에서, 가장 확실)
^C

# 방법 2 — q 키
# "Jetson Local View" 창을 마우스로 클릭하여 포커스 이동 후 q 입력
# ※ 터미널에서 q를 눌러도 동작하지 않음 (cv2 창 포커스 필요)
```

**종료 시 자동 처리:**
- `motp_log.csv` 저장 (`--motp --motp-log on` 시)
- `trajectory_log.csv` 저장 (`--traj-log on` 시)
- RealSense 파이프라인 정상 종료
- GStreamer 스트림 정상 종료
- CUDA GPU 메모리 해제

---

*본 매뉴얼은 `jetson_inference_advanced.py` 커밋 `aa88eb6` 기준으로 작성되었습니다.*
