# jetson_inference_advanced.py — 내부 흐름 기술 매뉴얼

**파일:** `drone_precision_landing/jetson_inference_advanced.py`  
**목적:** Jetson Nano 기반 드론 정밀 착륙 비전 시스템  
**작성일:** 2026-06-03

---

## 목차

1. [시스템 전체 아키텍처](#1-시스템-전체-아키텍처)
2. [시작 초기화 흐름](#2-시작-초기화-흐름)
3. [메인 루프 처리 흐름](#3-메인-루프-처리-흐름)
4. [컴포넌트별 내부 동작](#4-컴포넌트별-내부-동작)
5. [소스 우선순위 로직](#5-소스-우선순위-로직)
6. [데이터 흐름 맵](#6-데이터-흐름-맵)
7. [상태 기계 (State Machine)](#7-상태-기계-state-machine)
8. [메모리 구조](#8-메모리-구조)
9. [성능 타이밍 분석](#9-성능-타이밍-분석)
10. [CLI 인수별 동작 변화](#10-cli-인수별-동작-변화)

---

## 1. 시스템 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Jetson Nano (CPU + Maxwell GPU)                   │
│                                                                     │
│  하드웨어 입력                                                       │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐   │
│  │ RealSense    │    │           메인 처리 파이프라인              │   │
│  │ D435i        │───▶│                                          │   │
│  │ 컬러: 640×480│    │  EIS ──▶ YOLO ──▶ ArUco ──▶ NMS         │   │
│  │ 깊이: 640×480│    │                                          │   │
│  │ 30fps        │    │  트래커 ──▶ 칼만 필터 ──▶ MAVLink 송신   │   │
│  └──────────────┘    │                                          │   │
│                      │  LSTM 예측 ──▶ 화면 시각화               │   │
│  GPU 모델            └──────────────────────────────────────────┘   │
│  ┌──────────────┐                                                   │
│  │ best.engine  │    하드웨어 출력                                   │
│  │ YOLOv8       │    ┌───────────┐   ┌───────────┐                  │
│  │ TensorRT     │    │ MAVLink   │   │ GStreamer  │                  │
│  │ FP32/FP16    │    │ Pixhawk   │   │ UDP 스트림 │                  │
│  └──────────────┘    │ FC 착륙   │   │ GCS PC    │                  │
│                      └───────────┘   └───────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트 관계

```
JetsonTRTEngine ──── YOLO 추론 (GPU)
       │
       ▼
    raw_out (8400 앵커)
       │
       ▼
  NMSBoxes ──────── 중복 제거
       │
       ▼
  ┌────┴────┐
  │         │
single   ByteTracker ──── 다중 ID 추적
  │         │
  └────┬────┘
       │
  3D KalmanFilter ── 위치/속도 추정
       │
       ▼
  LSTMPredictor ─── 미래 궤적 예측
       │
  MOTPEvaluator ─── 추적 정밀도 측정
       │
  TrajectoryLogger ─ 궤적 CSV 저장
```

---

## 2. 시작 초기화 흐름

```
python3 jetson_inference_advanced.py [인수들]
│
├─[1] 인수 파싱 (argparse)
│     → 모든 설정값을 args 객체에 저장
│
├─[2] GStreamer 스트리밍 초기화 (--stream on 시)
│     ① nvv4l2h264enc (NVENC HW 인코더) 시도
│     ② 실패 시 x264enc (SW 인코더) 폴백
│     → cv2.VideoWriter 객체 생성
│
├─[3] MAVLink 연결
│     → mavutil.mavlink_connection(args.mav)
│     → recv_match(HEARTBEAT, timeout=3초)
│     → 타임아웃 시 master=None (영상만 처리)
│
├─[4] TensorRT 엔진 로드 (5~20초)
│     → best.engine 파일 역직렬화
│     → GPU 메모리 cudaMalloc (입/출력 버퍼)
│     → GPU 워밍업 추론 1회 (첫 프레임 JIT 지연 제거)
│
├─[5] RealSense D435i 초기화 (1~3초)
│     → 640×480 컬러 30fps
│     → 640×480 깊이 30fps
│     → 깊이-컬러 정렬 객체 (align)
│     → 깊이 필터 생성 (spatial, temporal)
│     → intrinsics 1회 취득 (세션 중 불변)
│
├─[6] ArUco 탐지기 초기화 (--aruco on 시)
│     ① cv2.aruco 가용성 확인 (contrib 필요)
│     ② cv2.aruco.ArucoDetector 또는 detectMarkers 선택
│     → 카메라 행렬 생성 (intrinsics 기반)
│
├─[7] 인스턴스 생성
│     → single_kf: 칼만 필터 (--model 선택)
│     → byte_tracker: ByteTracker (bytetrack 모드 시)
│     → track_3d_kfs: {} (트랙별 3D KF 딕셔너리)
│     → lstm_pred: LSTMPredictor (--predict>0 시)
│     → motp_eval: MOTPEvaluator (--motp 시)
│     → traj_log: TrajectoryLogger
│     → stabilizer: FrameStabilizer (--eis on 시)
│
└─[8] 메인 루프 시작
```

---

## 3. 메인 루프 처리 흐름

매 프레임(목표 30fps = 33.3ms)마다 다음 순서로 실행됩니다.

```
while True:
  │
  ├─[A] RealSense 프레임 수신
  │     pipeline.wait_for_frames()        ← 하드웨어 동기화 대기
  │     align.process()                    ← 깊이-컬러 픽셀 정렬
  │     get_depth_frame() / color_frame()
  │
  ├─[B] 깊이 후처리 필터 (2단계)
  │     spatial_filter  → 공간 노이즈 제거
  │     temporal_filter → 시간 노이즈 제거
  │     (hole_fill 제거: 5×5 median이 대체)
  │
  ├─[C] 컬러 이미지 준비
  │     _raw_view = asanyarray()           ← RS 버퍼 뷰 (복사 없음)
  │     EIS ON: color_image = stabilize()  ← 새 배열 반환
  │     EIS OFF: color_image = None        ← 추론 후 결정
  │
  ├─[D] YOLOv8 TRT 추론 (GPU, 15~25ms)
  │     blobFromImage() → 전처리 (1~2ms)
  │     cudaMemcpy H→D  → GPU 전송
  │     execute_v2()    → GPU 추론
  │     cudaMemcpy D→H  → 결과 회수
  │     raw_out(8400×5) → YOLO 앵커 출력
  │     EIS OFF: color_image = _raw_view.copy()  ← 추론 후 복사
  │
  ├─[E] ArUco 마커 탐지 (--aruco on 시)
  │     320×240 그레이로 축소
  │     detectMarkers() → 마커 ID, 코너 좌표
  │     ×2 스케일 복원 (float32 유지)
  │     estimatePoseSingleMarkers() → tvec(3D 오프셋)
  │     az 유효성 체크 (≥ 0.15m)
  │     → aruco_pose 튜플 설정 또는 None
  │
  ├─[F] YOLO 출력 처리
  │     score ≥ 0.3 필터링
  │     NMSBoxes(IoU 0.45) → 중복 제거
  │     all_dets 리스트 생성
  │
  ├─[G] 추적 모드 분기
  │     ┌─── single 모드 ────────────────────────────────┐
  │     │  소스 우선순위:                                │
  │     │  ① ArUco(aruco_pose≠None) → 3D KF 업데이트   │
  │     │  ② YOLO+depth → 5×5중앙값 → 역투영 → KF      │
  │     │  ③ YOLO+FOV각도 (depth 없을 때)               │
  │     └────────────────────────────────────────────────┘
  │     ┌─── bytetrack 모드 ──────────────────────────────┐
  │     │  ByteTracker.update(all_dets) → 활성 트랙 목록  │
  │     │  각 트랙 → 깊이 → 역투영 → 트랙별 3D KF        │
  │     │  주 트랙(최고 score) 선택                       │
  │     │  ArUco 보정 (150px 이내 시)                    │
  │     └─────────────────────────────────────────────────┘
  │
  ├─[H] 칼만 필터 처리
  │     predict() → 이전 상태로 현재 예측
  │     correct() → 측정값으로 보정
  │     → fx,fy,fz(위치), vx,vy,vz(속도), innov(혁신)
  │
  ├─[I] LSTM 미래 예측 (--predict N 시)
  │     buf에 현재 위치 추가
  │     15프레임마다 온라인 학습
  │     predict() → N스텝 미래 위치 목록
  │
  ├─[J] MAVLink 송신 (10Hz 제한)
  │     angle_x = atan2(fx, fz)
  │     angle_y = atan2(fy, fz)
  │     landing_target_send() → Pixhawk FC
  │
  ├─[K] 화면 시각화
  │     draw_info(5, 93, ...) → 위치/속도 텍스트 (고정 위치)
  │     cv2.rectangle/circle → 바운딩박스/중심
  │     LSTM 점 → 미래 궤적
  │     MOTP 텍스트 (--motp 시)
  │
  ├─[L] 평가 및 로깅
  │     motp_eval.update(innov) → MOTP 누적
  │     traj_log.update(...)    → 궤적 로그
  │
  ├─[M] 출력
  │     GStreamer out.write() (--stream on 시)
  │     cv2.imshow("Jetson Local View", color_image)
  │     cv2.waitKey(1) → 'q' 감지
  │
  └─[N] 소멸 트랙 정리 (bytetrack 시)
        track_3d_kfs 미활성 트랙 삭제
        _color_cache 함께 제거
```

---

## 4. 컴포넌트별 내부 동작

### 4.1 JetsonTRTEngine — GPU 추론 엔진

```
초기화:
  best.engine 파일 읽기
      → deserialize_cuda_engine()
      → create_execution_context()
  각 바인딩별:
      host_mem = np.empty(shape)    ← CPU 메모리
      cudaMalloc(cuda_ptr)          ← GPU 메모리
      bindings 리스트에 추가

infer(img) 호출 시:
  ① blobFromImage(img, 1/255.0, (640,640))
     → (1, 3, 640, 640) float32 배열 (C++ 단일 패스)
  ② np.copyto(host_buf, blob)
     → CPU 호스트 버퍼에 복사
  ③ cudaMemcpy(device, host, H2D)
     → GPU VRAM으로 전송
  ④ context.execute_v2(bindings)
     → GPU 추론 실행 (15~25ms)
  ⑤ cudaMemcpy(host, device, D2H)
     → 결과를 CPU로 회수
  반환: self.outputs[0]['host']  ← 내부 버퍼 View
```

**주의:** 반환값은 내부 버퍼 View → 다음 infer() 전 처리 완료 필수

---

### 4.2 칼만 필터 4모델

#### CV (등속도 6D)
```
상태: [x, y, z, vx, vy, vz]
전이: x += vx·dt  (속도 일정 가정)
적용: 직선 이동, 정적 패드
```

#### CA (등가속도 9D)
```
상태: [x, y, z, vx, vy, vz, ax, ay, az]
전이: x += vx·dt + 0.5·ax·dt²
      vx += ax·dt
적용: 가감속 기동
```

#### CTRV (선회율 7D EKF)
```
상태: [x, y, z, v, yaw, yaw_rate, vz]
전이 (비선형):
  |yaw_rate| < ε → 직선: x += v·cos(yaw)·dt
  |yaw_rate| ≥ ε → 선회: x += (v/ω)·(sin(yaw+ω·dt)-sin(yaw))
야코비안 F 계산 후 EKF Predict/Correct
Joseph 안정형 P 업데이트
적용: 드론 선회 기동
```

#### IMM (CV+CTRV 확률 혼합)
```
μ[CV]=0.5, μ[CTRV]=0.5 초기값
매 프레임:
  L[i] = exp(-0.5·(innov/σ)²)   ← 우도 계산
  μ = L·(Π·μ) / Σ               ← 확률 갱신
  output = Σ μ[i]·result[i]      ← 가중 평균
적용: 복합 기동 (직선↔선회 자동 전환)
```

---

### 4.3 ByteTracker — 3단계 IoU 매칭

```
update(all_dets) 호출 시:

  high = [d for d if score ≥ 0.6]    ← 고신뢰도
  low  = [d for d if 0.3 ≤ score]    ← 저신뢰도

  모든 트랙 predict()                  ← KalmanBoxFilter 예측

  1차 매칭: tracked ↔ high (IoU ≥ 0.7)
    → 헝가리안 알고리즘 (scipy) 또는 그리디 폴백
    → 매칭: track.update(det)
    → 미매칭 트랙: → 2차로

  2차 매칭: 미매칭 tracked ↔ low (IoU ≥ 0.5)
    → 매칭: track.update(det)
    → 미매칭: state='lost'

  3차 매칭: lost ↔ 남은 고신뢰도 (재진입)
    → 매칭: 복귀 (reactivated)
    → 미매칭: 새 STrack 생성 (새 ID)

  lost 트랙 age 관리 (max_lost=15프레임)
  
  반환: tracklet_len ≥ 2인 confirmed 트랙만
```

**STrack 내부 상태:**
```
new → tracked (2프레임 확인) → lost → [max_lost 초과 시 삭제]
                                      ← 재매칭 시 복귀
```

---

### 4.4 LSTMPredictor — 온라인 학습 궤적 예측

```
구조: LSTM(input=3, hidden=64, layers=2) → Linear(64, N×3)
      입력: (batch, 30, 3)  ← 과거 30프레임 xyz
      출력: (batch, N, 3)   ← 미래 N프레임 xyz

add(x, y, z):
  self.ref 기준 정규화 후 buf에 추가
  15프레임마다 _train() 호출

_train():
  슬라이딩 윈도우로 (입력,타겟) 쌍 생성
  Adam 옵티마이저, MSE 손실, 5 에폭
  50ms 초과 시 프레임 드롭 경고

predict():
  buf < 30: None 반환
  미학습: 선형 외삽 (최근 5프레임 기울기)
  학습됨: LSTM 추론 → 미래 N개 xyz 반환
```

---

### 4.5 FrameStabilizer — EIS 영상 안정화

```
stabilize(frame) 호출 시:

  gray = cvtColor(frame, BGR2GRAY)

  ① 초기화: _prev_gray 없으면 저장 후 원본 반환
  ② goodFeaturesToTrack(_prev_gray)
     → 최대 200개 코너 특징점
  ③ calcOpticalFlowPyrLK(prev, curr, pts)
     → 특징점 추적 (21×21 윈도우, 3레벨)
  ④ estimateAffinePartial2D(RANSAC)
     → dx, dy, da (이동+회전) 추정
     → inliers < 8: 원본 반환
  ⑤ 누적 궤적 갱신 (_cum_dx += dx 등)
     이동 평균 스무딩 (deque maxlen=smoothing)
     smoothing 미충족: 원본 반환
  ⑥ diff = smooth - cum  ← 제거할 진동 성분
  ⑦ warpAffine(frame, M, BORDER_REPLICATE)
     → 안정화된 프레임 반환
```

---

### 4.6 ArUco 탐지 흐름

```
매 프레임:
  color_image → cvtColor → gray_full (640×480)
  gray_full → resize → gray_half (320×240)
  
  _aruco_detect(gray_half):
    → corners_half (N개 마커 코너 좌표, 320×240 공간)
    → ids (마커 ID 배열)

  corners_half × 2 → 640×480 공간으로 복원
    (c * 2).astype(float32)  ← float32 명시 유지

  az 유효성 체크 (≥ 0.15m):
    유효: drawDetectedMarkers + drawFrameAxes
          aruco_pose = (ax, ay, az, rvec, tvec, mid, cx_a, cy_a)
    무효: drawDetectedMarkers만 (경고 없음)
          aruco_pose = None → YOLO 폴백
```

---

## 5. 소스 우선순위 로직

### Single 모드

```
프레임 시작
    │
    ├─ aruco_pose ≠ None?
    │       ↓ YES
    │   ArUco tvec → KF → 각도 계산
    │   MAVLink 송신 (atan2 기반)
    │
    ├─ YOLO best score ≥ 0.6?
    │       ↓ YES
    │   get_median_depth(cx, cy) → 유효?
    │       ↓ YES                    ↓ NO
    │   rs2_deproject → KF       FOV 각도만
    │   MAVLink 송신              MAVLink 송신
    │
    └─ 감지 없음
        KF.reset()
        LSTM.reset()
```

### ByteTrack 모드

```
ByteTracker.update(all_dets)
    │
    └─ active 트랙 목록 (confirmed만)
        │
        for each track:
            get_median_depth(cx, cy)
            depth 유효 → rs2_deproject → track_3d_kfs[tid].update()
            depth 무효 → FOV 각도만
            score 최고 → primary 업데이트
        │
        primary 결정 후:
            ArUco pose 150px 이내 → 각도 보정 (KF 재호출 없음)
            draw_info(5, 93, ...) ← 고정 위치
            LSTM 예측 (주 트랙만)
            MAVLink 송신
```

---

## 6. 데이터 흐름 맵

```
RealSense HW
  ├─ depth_frame ──→ spatial_filter ──→ temporal_filter
  │                                           │
  │                            get_median_depth(cx, cy)
  │                            rs2_deproject_pixel_to_point()
  │                                           │
  └─ color_frame ──→ _raw_view ──→ [EIS] ──→ color_image
                                       │
                              blobFromImage()
                                       │
                              GPU TRT infer()
                                       │
                              raw_out (8400×5)
                                       │
                              NMSBoxes → all_dets
                                       │
              ┌────────────────────────┤
              │                        │
          single 모드           bytetrack 모드
              │                        │
         [ArUco pose]          ByteTracker.update()
              │                        │
         KalmanFilter          track_3d_kfs[tid]
         .update(x,y,z)        .update(x,y,z)
              │                        │
         fx,fy,fz              fx,fy,fz (primary)
         vx,vy,vz              vx,vy,vz
         innov                 innov
              │                        │
              └────────────┬───────────┘
                           │
                  ┌────────┼────────┐
                  │        │        │
              MAVLink   LSTM     MOTP/Traj
              송신      예측     로그
              10Hz      N스텝    저장
```

---

## 7. 상태 기계 (State Machine)

### 칼만 필터 상태

```
[초기화 안됨]
    │ 첫 측정값 입력
    ▼
[추적 중]
    │ 연속 감지
    ▼
[리셋 대기]   ← 감지 실패 시 KF.reset() 호출
    │ 다음 감지 시 재초기화
    ▼
[추적 중] (재시작)
```

### ByteTrack STrack 상태

```
[신규 new]
    │ tracklet_len ≥ 2
    ▼
[확인됨 tracked] ←─── 재매칭 복귀
    │ 매칭 실패
    ▼
[소실 lost]
    │ max_lost(15프레임) 초과
    ▼
[삭제]
```

### ArUco 감지 상태

```
[감지 안됨]
    │ ArUco 마커 발견 + az ≥ 0.15m
    ▼
[유효 aruco_pose 설정]
    │ 칼만 필터에 tvec 입력
    ▼
[YOLO 폴백] ← az < 0.15m 또는 감지 실패
```

### LSTM 상태

```
[데이터 수집 중]
    │ buf 크기 ≥ SEQ_LEN + steps
    ▼
[학습 가능] (매 15프레임마다 학습)
    │ ready=True
    ▼
[예측 활성] → predict() 반환 N개 미래 좌표
    │ 트랙 변경 또는 리셋
    ▼
[리셋] → buf.clear(), ref=None
```

---

## 8. 메모리 구조

### GPU 메모리 (고정 할당)

```
TRT 입력 버퍼: (1, 3, 640, 640) × float32 = 4.9MB
TRT 출력 버퍼: (1, 5, 8400) × float32    = 168KB
합계: ~5.1MB GPU VRAM (cudaMalloc, 프로그램 수명 동안 유지)
```

### CPU 메모리 (동적)

```
영구 고정:
  RealSense 내부 버퍼: ~90MB (3중 버퍼, depth+color)
  stabilizer._prev_gray: 300KB (640×480 gray)
  lstm_pred.net 가중치: ~100KB (PyTorch)

동적 상한 있음:
  lstm_pred.buf: 120 × 3 × float64 = 2.9KB
  motp_eval._log: 최대 18,000행 × 20B = 360KB
  traj_log._log: 최대 54,000행 × 80B = 4.3MB
  track_3d_kfs: 트랙수 × ~1KB
  _color_cache: 트랙수 × 3int (미미)
  stabilizer._traj: smoothing × 3float = 168B

프레임당 임시 (GC 즉시 해제):
  _raw_view: 0B (RS 버퍼 View)
  blobFromImage: 4.9MB (TRT 전처리)
  color_image: 900KB (.copy() 시)
  gray_half: 75KB (320×240)
```

---

## 9. 성능 타이밍 분석

### 각 단계별 소요 시간

```
단계                       시간      비고
───────────────────────────────────────────────────
① wait_for_frames()       2~5ms    하드웨어 동기
② align.process()         0.5ms    깊이-컬러 정렬
③ 깊이 필터 2단계         2~3ms    spatial+temporal
④ blobFromImage 전처리    1~2ms    ← blobFromImage 최적화 적용
⑤ cudaMemcpy H→D         0.5ms    입력 전송
⑥ execute_v2 (TRT)       15~25ms  GPU 추론 (최대 병목)
⑦ cudaMemcpy D→H         0.1ms    결과 회수
⑧ color_image.copy()     0.5ms    (EIS OFF 시)
⑨ ArUco 320×240          1~3ms    (--aruco on 시)
⑩ NMSBoxes               0.3ms    중복 제거
⑪ ByteTrack or KF        0.5ms    추적 처리
⑫ 화면 그리기            0.5ms    putText/rectangle
⑬ GStreamer NVENC        0.3ms    HW 인코딩
   GStreamer x264         2~5ms    SW 인코딩
⑭ imshow + waitKey       1~2ms    화면 표시
⑮ print (verbose off)    0ms      ← 제거됨
───────────────────────────────────────────────────
합계 (기본, EIS OFF)     27~43ms  30fps 경계선
합계 (EIS ON, +5~8ms)   32~51ms  30fps 초과 위험
```

### 30fps 달성 가능 조합

| 구성 | 합계 | 30fps |
|------|------|-------|
| 기본 (cv, single) | ~22ms | ✅ |
| +ArUco, NVENC | ~27ms | ✅ |
| +EIS smoothing=5 | ~32ms | ⚠️ |
| +EIS smoothing=7 | ~35ms | ❌ |

---

## 10. CLI 인수별 동작 변화

### `--model` — 칼만 필터 모델

| 값 | 상태 차원 | 연산 | 적합 상황 |
|----|---------|------|---------|
| `cv` | 6D | 선형 KF | 기본, 직선 이동 |
| `ca` | 9D | 선형 KF | 가감속 |
| `ctrv` | 7D | EKF+야코비안 | 드론 선회 |
| `imm` | CV+CTRV 혼합 | 확률 가중 | 복합 기동 |

### `--tracker` — 추적 방식

```
single:
  - 8400 앵커 중 score 최고 1개만 추적
  - track_3d_kfs 없음 → single_kf만 사용
  - 단순하고 빠름

bytetrack:
  - 3단계 IoU 매칭으로 다중 트랙 유지
  - track_3d_kfs[tid]: 트랙별 독립 3D KF
  - 주 트랙(score 최고)만 MAVLink 송신
  - 재진입 처리 (ID 유지)
```

### `--aruco on` — ArUco 우선순위

```
ArUco 탐지 성공 (az ≥ 0.15m):
  → tvec 직접 3D 위치 사용
  → 깊이 센서 불필요
  → 서브픽셀 정밀도
  → 6DOF 자세 추정 가능

ArUco 실패:
  → YOLO + depth 폴백
  → YOLO + FOV 각도 (depth 없을 때)
```

### `--eis on` — 영상 안정화

```
적용 범위: YOLO 추론 + ArUco 탐지 + 화면 표시
           (depth_frame은 원본 좌표 유지)

효과: 드론 진동 제거 → YOLO score 향상
비용: 5~8ms/frame 추가 → 30fps 달성 어려움

깊이 오차: shift_px × depth_m × 0.00188
           (예: 10px, 2m → 3.8cm, 허용 범위)
```

### `--predict N` — LSTM 미래 예측

```
N=0: LSTMPredictor 미생성 (비용 0)
N>0: 
  - 45프레임(=SEQ_LEN+steps) 수집 후 예측 시작
  - 15프레임마다 온라인 학습 (50~200ms 블로킹)
  - 예측 결과 → 화면에 점(N개)으로 표시
  - ByteTrack: 주 트랙 ID 변경 시 buf 리셋
```

### `--verbose on/off`

```
off (기본): 루프 내 print 없음
  → 0.4~2.6ms/frame 절약 (SSH 환경에서 효과 큼)
  → 오류/경고 print는 항상 출력

on: 매 프레임 탐지 결과 출력
  → 🎯 위치/속도 정보
  → 📡 MAVLink 송신 각도/거리
```

---

## 부록 — 화면 레이아웃

```
640×480 프레임
┌─────────────────────────────────────────┐
│ SRC:ArUco ID:0 Z:2.45m|CTRV[Line]  ←y=20
│ Offset X:-0.04m  Y:-0.07m          ←y=38
│ Alt(Z): 2.45m                       ←y=53
│ Vel Vx:0.03 Vy:-0.01 Vz:-0.12 m/s ←y=68
│ Speed:0.13m/s [13cm/s]              ←y=83
│                                     │
│        ┌────────────┐               │
│        │(ArUco 윤곽)│               │
│        │   ○ 원     │               │
│        │  /X Y Z축/ │               │
│        └────────────┘               │
│                                     │
│   ● ● ●  (LSTM 미래 예측 점)        │
│                                     │
│ MOTP:2.34cm (N=150)             ←y=470
└─────────────────────────────────────────┘
```

---

*본 매뉴얼은 `jetson_inference_advanced.py` 커밋 `aa88eb6` 기준으로 작성되었습니다.*
