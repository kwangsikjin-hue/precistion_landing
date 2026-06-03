# jetson_inference_advanced.py — 수정 이력 및 변경 매뉴얼

**파일:** `drone_precision_landing/jetson_inference_advanced.py`  
**원본:** `drone_precision_landing/jetson_inference.py`  
**최종 커밋:** `9265db9`  
**작성일:** 2026-06-03

---

## 목차

1. [전체 수정 이력 요약](#1-전체-수정-이력-요약)
2. [커밋별 상세 수정 내용](#2-커밋별-상세-수정-내용)
3. [카테고리별 수정 항목 전체 목록](#3-카테고리별-수정-항목-전체-목록)
4. [원본 대비 주요 기능 추가 목록](#4-원본-대비-주요-기능-추가-목록)
5. [전체 CLI 인수 목록](#5-전체-cli-인수-목록)
6. [환경별 실행 예시](#6-환경별-실행-예시)
7. [알려진 이슈 및 제약 사항](#7-알려진-이슈-및-제약-사항)

---

## 1. 전체 수정 이력 요약

| 커밋 | 분류 | 수정 건수 | 내용 요약 |
|------|------|-----------|-----------|
| `8a4c133` | feat | 신규 생성 | 원본 기반 고도화 시스템 최초 작성 |
| `a7e80b1` | fix | 5건 | ByteTrack 논리 버그 수정 |
| `9baaeb8` | fix | 8건 | 통신·메모리·수치 안정성 버그 수정 |
| `edfba97` | fix | 1건 | MAVLink 송출 print 복원 |
| `31c6ca5` | fix | 3건 | jetson_inference_modify.py 동일 결함 적용 |
| `d630197` | fix | 2건 | `--stream on` 실행 시 get_distance() 타입 오류 수정 |
| `b0a37ba` | perf | 5건 | 시작 지연 원인 개선 (진행 메시지·GPU 워밍업·타임아웃 단축) |
| `e670f34` | feat | 2건 | `--mav`, `--mav-timeout` CLI 인수 추가 |
| `2213c23` | fix | 2건 | `cv2.putText` 한글 → ASCII 교체 (`?????` 표시 수정) |
| `9265db9` | feat | 6건 | ArUco 마커 탐지기 통합 (`--aruco`, `--aruco-dict`, `--marker-size`) |

**총 수정·개선: 34건**

---

## 2. 커밋별 상세 수정 내용

---

### 커밋 `8a4c133` — 최초 고도화 버전 생성

원본 `jetson_inference.py`를 기반으로 다음 기능들을 추가하여 신규 작성:

#### 추가된 CLI 인수

| 인수 | 값 | 설명 |
|------|----|------|
| `--stream` | `on` / `off` | GStreamer UDP 스트리밍 (원본 기능 유지) |
| `--model` | `cv`, `ca`, `ctrv`, `imm` | 칼만 필터 모델 선택 (신규) |
| `--tracker` | `single`, `bytetrack` | 추적 방식 선택 (신규) |
| `--predict` | 정수 N | LSTM 미래 예측 스텝 (신규) |
| `--motp` | 플래그 | MOTP 정밀도 평가 (신규) |

#### 추가된 클래스

| 클래스 | 설명 |
|--------|------|
| `KFModelCV` | 등속도 6D 선형 칼만 필터 |
| `KFModelCA` | 등가속도 9D 선형 칼만 필터 |
| `KFModelCTRV` | 선회율 7D 비선형 EKF |
| `KFModelIMM` | CV+CTRV 확률 혼합 IMM |
| `KalmanBoxFilter` | ByteTrack용 2D 바운딩박스 칼만 |
| `STrack` | ByteTrack 단일 트랙 객체 |
| `ByteTracker` | 3단계 IoU 매칭 다중 객체 추적기 |
| `LSTMPredictor` | 온라인 학습 LSTM 미래 궤적 예측기 |
| `MOTPEvaluator` | Innovation 기반 추적 정밀도 평가기 |

#### 원본 대비 개선된 하드웨어 처리

- `intrinsics` 루프 밖 1회 취득 (원본: 매 프레임 취득)
- RealSense 깊이 후처리 필터 체인 추가 (`spatial` → `temporal` → `hole_fill`)
- `get_median_depth()` 5×5 중앙값 깊이 측정 (원본: 단일 픽셀)
- 유효 깊이 범위 체크: `DEPTH_MIN_M=0.15` ~ `DEPTH_MAX_M=8.0`
- `cudaFree` `__del__` 추가로 CUDA 메모리 누수 방지

#### 원본 고유 기능 보존

- `--stream on/off` GStreamer 조건부 활성화
- `udpin:0.0.0.0:14551` MAVLink 연결
- FOV 기반 각도 폴백: 깊이 없어도 `angle_x/y` 계산 후 MAVLink 송신
- MAVLink 1/2 호환 (`try/except TypeError`)
- `KeyboardInterrupt` + `Exception` + `finally` 종료 구조

---

### 커밋 `a7e80b1` — ByteTrack 논리 버그 5건 수정

#### B1 — `_associate()` 반환값 오류 ★ 가장 심각

**문제:** `tracks=[]` 또는 `dets=[]` 일 때 unmatched 목록 반환값이 잘못됨

```python
# 수정 전: 복잡한 조건식으로 반환값 일부가 잘못된 목록 반환
return [],[list(range(len(tracks))),list(range(len(dets)))][0 if not tracks else 1],\
       [list(range(len(dets))),list(range(len(tracks)))][0 if not dets else 1]

# 수정 후: 명확한 3-tuple 반환
return [], list(range(len(tracks))), list(range(len(dets)))
```

**영향:** ByteTrack 시작 시 첫 프레임에서 모든 검출이 신규 트랙으로 등록되지 않음

---

#### B2 — 3차 매칭 재진입 트랙이 `self.tracked`에 추가되지 않는 버그

**문제:** `lost → tracked` 재진입 트랙이 `self.lost` 갱신 후 사라짐

```python
# 수정 전: self.lost에서 재매칭 트랙을 제거한 뒤 다시 추가하려 했으나 이미 사라짐
self.lost = [t for t in self.lost+newly_lost if ... and t.state=='lost']
self.tracked = [...] + [t for t in self.lost if t.state=='tracked']  # 항상 빈 리스트

# 수정 후: 재매칭 트랙을 reactivated 리스트로 먼저 수집
reactivated = []
for ti, di in m3:
    self.lost[ti].update(rem_high[di], self.frame_id)
    reactivated.append(self.lost[ti])   # state='tracked'로 변경
self.tracked = [...] + reactivated + new_tracks
```

**영향:** 화면 밖으로 나갔다 돌아온 객체가 재추적되지 않고 ID가 소멸

---

#### B3 — `LSTMPredictor.predict()` `self.ref` None 안전장치

```python
# 수정 전
if len(self.buf) < self.SEQ_LEN: return None

# 수정 후
if len(self.buf) < self.SEQ_LEN or self.ref is None: return None
```

---

#### B4 — `track_color()` 전역 난수 상태 오염

```python
# 수정 전: np.random.seed()로 전역 난수 상태 변경
np.random.seed(tid * 17 % 256)
return tuple(int(c) for c in np.random.randint(80, 255, 3))

# 수정 후: 독립 RNG 인스턴스 사용
rng = np.random.RandomState(tid * 17 % 256)
return tuple(int(c) for c in rng.randint(80, 255, 3))
```

---

#### B5 — ByteTrack primary 깊이 없을 때 `0` 값 위치 출력 혼동

```python
# 수정 후: 깊이 유효 여부에 따라 출력 분기
pos_str = (f"X:{fx*100:.1f} Y:{fy*100:.1f} Z:{fz*100:.1f}cm"
           if dv > 0 else "깊이미확정(FOV각도로송신)")
```

---

### 커밋 `9baaeb8` — 통신·메모리·수치 안정성 8건 수정

#### C1 — RealSense 깊이 필터 후 타입 변환 누락

**문제:** `hole_fill.process()` 후 `as_depth_frame()` 미적용 → `get_distance()` `AttributeError`

```python
# 수정 전
depth_frame = hole_fill.process(depth_frame)

# 수정 후
depth_frame = hole_fill.process(depth_frame).as_depth_frame()
```

---

#### C2 — `send_mavlink()` 네트워크 예외 미처리

**문제:** MAVLink 1/2 `TypeError`만 잡고 소켓 오류(`OSError`) 미처리 → 메인루프 크래시

```python
# 수정 후: 각 경로에 Exception 처리 추가
except TypeError:
    try:
        master.mav.landing_target_send(...)
    except Exception as mav_err:
        print(f"⚠️ MAVLink 송신 실패 (MAVLink1): {mav_err}")
        return
except Exception as mav_err:
    print(f"⚠️ MAVLink 송신 실패 (MAVLink2): {mav_err}")
    return
```

---

#### C3 — `KFModelCTRV` S 특이행렬 → `LinAlgError` 크래시

**문제:** `np.linalg.inv(S)` — S가 특이행렬이면 비정상 종료

```python
# 수정 전
K = P_pred @ self.H.T @ np.linalg.inv(S)
self.P = (np.eye(7) - K @ self.H) @ P_pred

# 수정 후: solve로 교체 + Joseph 안정형 P 업데이트
K = P_pred @ self.H.T @ np.linalg.solve(S.T, np.eye(self.m)).T
IKH = np.eye(7) - K @ self.H
self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
```

---

#### C4 — `KalmanBoxFilter` h≈0 시 S 특이행렬

**문제:** 박스 높이(h)가 0에 가까우면 S가 특이행렬로 크래시

```python
# 수정 후
h = max(mean[3], 1.0)   # h≈0 방지
K = cov @ H.T @ np.linalg.solve(S.T, np.eye(4)).T
IKH = np.eye(8) - K @ H
new_cov = IKH @ cov @ IKH.T + K @ R @ K.T   # Joseph 형식
```

---

#### C5 — `MOTPEvaluator._log` 무제한 메모리 성장

**문제:** 장시간 실행 시 `_log` 리스트가 무한 증가

```python
# 수정 후: 18,000건(30fps×10분) 초과 시 앞 절반 자동 삭제
_LOG_MAX = 18000
if len(self._log) > self._LOG_MAX:
    self._log = self._log[self._LOG_MAX // 2:]
```

---

#### C6 — LSTM 학습이 메인 루프를 동기 블로킹

**문제:** `_train()` 호출이 메인 루프에서 50~200ms 블로킹 가능

```python
# 수정 후: 학습 시간 측정 및 경고 출력
t0 = time.time()
for _ in range(self.EPOCHS):
    ...
elapsed = (time.time() - t0) * 1000
if elapsed > 50:
    print(f"⚠️ [LSTM] 학습 {elapsed:.0f}ms — 프레임 드롭 주의")
```

---

#### C7 — single 모드 `cx,cy` 경계 밖 시 MAVLink 미송신

**문제:** `cx,cy`가 이미지 경계 밖이면 MAVLink 미송신

```python
# 수정 후: 클램프 후 항상 처리
cx_c = max(0, min(639, cx))
cy_c = max(0, min(479, cy))
angle_x, angle_y = fov_angles(cx_c, cy_c)
dv = get_median_depth(depth_frame, cx_c, cy_c)
# ... MAVLink 항상 송신
```

---

#### C8 — 들여쓰기 오류 (`if True:` 제거 후 블록 재정렬)

`C7` 적용 과정에서 발생한 들여쓰기 파싱 오류 수정

---

### 커밋 `edfba97` — MAVLink 송출 print 복원

**문제:** `C2` 예외처리 추가 시 `send_mavlink()` 내 print가 실수로 삭제됨

```python
# 복원된 print (송신 성공 시)
print(f"📡 [MAVLink] LANDING_TARGET 송출 -> "
      f"angle_x:{math.degrees(angle_x):.2f}°, "
      f"angle_y:{math.degrees(angle_y):.2f}°, "
      f"Dist:{distance:.2f}m")
```

---

### 커밋 `31c6ca5` — `jetson_inference_modify.py` 동일 결함 3건 적용

#### M4 — Heartbeat 타임아웃 시 `master = None` 미설정

**문제:** 타임아웃 시 `master` 객체가 살아있어 의미없는 MAVLink 패킷 지속 송신

```python
# 수정 전
else:
    print("⚠️ [경고] 5초간 Heartbeat 없음 — 영상 처리만 강행합니다.")
    # master는 None이 아닌 상태로 남음!

# 수정 후
else:
    print("⚠️ [경고] 5초간 Heartbeat 없음 — master=None 처리 후 영상 처리만 진행합니다.")
    master = None
```

---

#### C2(원본 분류) — `cudaMalloc` 반환 코드 미확인

**문제:** GPU 메모리 부족 시 NULL 포인터로 이후 `cudaMemcpy`에서 segfault

```python
# 수정 전
self.cuda_lib.cudaMalloc(ctypes.byref(cuda_ptr), host_mem.nbytes)

# 수정 후
ret = self.cuda_lib.cudaMalloc(ctypes.byref(cuda_ptr), host_mem.nbytes)
if ret != 0:
    raise RuntimeError(f"cudaMalloc 실패 (에러코드: {ret}) — GPU 메모리 부족 가능성")
```

---

#### M1 — Python `for` 루프 8400개 순회 → numpy 벡터화

**문제:** Python for 루프로 8400개 박스를 순회하면 Jetson Nano에서 5~15ms/frame 지연

```python
# 수정 전 (5~15ms/frame)
for pred in raw_out:
    score = float(pred[4])
    if score < 0.3: continue
    xc, yc, w, h = pred[:4]
    ...
    all_dets.append([x1, y1, x2, y2, score])

# 수정 후 (~0.1ms/frame)
scores  = raw_out[:, 4]
vmask   = scores >= 0.3
vpreds  = raw_out[vmask]
if len(vpreds):
    x1s = (xc - bw/2).astype(int)
    ...
    all_dets = [[int(x1s[i]),...,float(sc[i])] for i in range(len(vpreds))]
```

---

### 커밋 `d630197` — `--stream on` 실행 시 `get_distance()` 타입 오류 2건

**에러 메시지:**
```
get_distance(): incompatible function arguments
  지원 형식: (self: pyrealsense2.depth_frame, x: int, y: int)
  실제 호출: (<pyrealsense2.frame Z16 #20>, 247.0, 185.0)
```

#### 원인 ① — 필터 체인 단계별 타입 손실

**문제:** `spatial/temporal` 필터 출력이 `rs.frame`으로 타입 변환되어 `get_distance()` 호출 불가

```python
# 수정 전: 마지막 단계에만 변환 → 중간 단계에서 타입 손실
depth_frame = spatial_filter.process(depth_frame)           # rs.frame (타입 손실)
depth_frame = temporal_filter.process(depth_frame)          # rs.frame (타입 손실)
depth_frame = hole_fill.process(depth_frame).as_depth_frame()  # 너무 늦음

# 수정 후: 매 단계 명시 변환
depth_frame = spatial_filter.process(depth_frame).as_depth_frame()
depth_frame = temporal_filter.process(depth_frame).as_depth_frame()
depth_frame = hole_fill.process(depth_frame).as_depth_frame()
```

---

#### 원인 ② — M1 벡터화에서 좌표 `float` 업캐스팅

**문제:** `np.column_stack([x1s, y1s, x2s, y2s, sc])` 시 `sc`(float32)로 좌표 열 전체가 float 변환 → `get_distance(247.0, 185.0)` 타입 오류

```python
# 수정 전: column_stack으로 float 업캐스팅
all_dets = np.column_stack([x1s, y1s, x2s, y2s, sc]).tolist()
# → [[3.0, 10.0, 247.0, 185.0, 0.87], ...]  ← 좌표가 float!

# 수정 후: int/float 명시 분리
all_dets = [
    [int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i]), float(sc[i])]
    for i in range(len(vpreds))
]
# → [[3, 10, 247, 185, 0.87], ...]  ← 좌표가 int ✓
```

**안전망 추가:** `get_median_depth()` 내부에서도 `int()` 강제 캐스팅

```python
def get_median_depth(depth_frame, cx, cy, half=2):
    cx, cy = int(cx), int(cy)   # 어떤 타입이 들어와도 int 보장
    ...
```

---

### 커밋 `b0a37ba` — 시작 지연 원인 개선 (perf)

#### 원인별 시작 소요 시간 분석

```
[GStreamer --stream on]  x264enc 초기화          ── 2~5초
[MAVLink]               Heartbeat 대기 (3초)     ── 1~3초  ← 5→3초 단축
[TensorRT ★가장 느림]   best.engine GPU 로드     ── 5~20초
[RealSense]             USB 장치+센서 워밍업      ── 1~3초
[GPU JIT]               첫 추론 커널 컴파일       ── 1~2초  ← 워밍업으로 제거
```

#### P1 — TensorRT 로딩 진행 메시지 + 소요시간 출력

```python
# 수정 전: 로딩 중 아무 출력 없음 → 프리즈처럼 보임
self.engine = runtime.deserialize_cuda_engine(f.read())

# 수정 후: 시작·완료·소요시간 출력
print(f"⏳ TensorRT 엔진 로딩 중: {engine_path} (5~20초 소요, 잠시 대기)")
t0 = time.time()
self.engine = runtime.deserialize_cuda_engine(f.read())
print(f"✅ TensorRT 엔진 로드 완료 ({time.time()-t0:.1f}초)")
```

---

#### P2 — GPU 워밍업 더미 추론 추가

**문제:** TensorRT 로드 후 **첫 번째 `execute_v2()`** 호출 시 GPU 커널 JIT 컴파일로 1~2초 지연

```python
# 수정 후: 빈 더미 이미지로 1회 추론 → 첫 실제 프레임 JIT 제거
print("⏳ GPU 워밍업 중 (첫 프레임 지연 제거)...")
_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
trt_brain.infer(_dummy)
print("✅ GPU 워밍업 완료")
```

---

#### P3 — RealSense / GStreamer 초기화 진행 메시지

```python
# RealSense
print("⏳ RealSense D435i 초기화 중...")
profile = pipeline.start(config)
print("✅ RealSense 초기화 완료")

# GStreamer
print("📡 [알림] GStreamer UDP 스트리밍 초기화 중... (x264enc 로딩, 2~5초 소요)")
out = cv2.VideoWriter(...)
if out.isOpened():
    print("✅ GStreamer 스트리밍 초기화 완료")
```

---

#### P4 — MAVLink Heartbeat 타임아웃 5초 → 3초 단축

```python
# 수정 전
msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)

# 수정 후 (--mav-timeout 인수로 조절 가능, 기본 3초)
msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=args.mav_timeout)
```

---

#### P5 — 전체 초기화 총 소요시간 표시

```python
_startup_begin = time.time()
# ... 모든 초기화 ...
_startup_sec = time.time() - _startup_begin
print(f"🚀 [성공] 전체 초기화 완료 — 총 소요시간: {_startup_sec:.1f}초")
```

---

### 커밋 `e670f34` — `--mav`, `--mav-timeout` CLI 인수 추가 (feat)

#### 배경 — `udpin:0.0.0.0` 사용 시 문제점

```
기존: udpin:0.0.0.0:14551 → 모든 인터페이스 수신 대기
      FC가 먼저 Heartbeat 보내야 연결 시작 (최대 5초 대기)

개선: 특정 IP 또는 udpout 지정으로 연결 방식 선택 가능
      udpout = Jetson이 FC에 먼저 연결 요청 → Heartbeat 즉시 수신
```

#### E1 — `--mav` 인수: MAVLink 연결 주소 설정

```python
# 추가된 인수
parser.add_argument('--mav', type=str, default='udpin:0.0.0.0:14551',
                    help='MAVLink 연결 주소\n'
                         '  udpout:192.168.0.10:14550  (SITL PC, 빠름)\n'
                         '  udpin:192.168.0.5:14551    (특정 인터페이스)\n'
                         '  /dev/ttyUSB0               (USB-UART 직렬)')
```

연결 방식별 특성:

| 방식 | 예시 | 속도 | 비고 |
|------|------|------|------|
| `udpin:0.0.0.0` | 기존 기본값 | 느림 | 모든 인터페이스, FC 먼저 Heartbeat |
| `udpin:특정IP` | `udpin:192.168.0.5:14551` | 보통 | 단일 인터페이스, 노이즈 감소 |
| `udpout:FC_IP` | `udpout:192.168.0.10:14550` | 빠름 | Jetson이 먼저 연결, 즉시 Heartbeat |
| `/dev/ttyUSBx` | `/dev/ttyUSB0` | 가장 빠름 | USB-UART 직렬, 지연 최소 |

---

#### E2 — `--mav-timeout` 인수: Heartbeat 대기 시간 설정

```python
parser.add_argument('--mav-timeout', type=int, default=3, metavar='SEC',
                    help='MAVLink Heartbeat 대기 타임아웃 초 (기본: 3)')
```

| 상황 | 권장값 |
|------|--------|
| FC 연결 확실 (실기체) | `1` |
| SITL 동시 시작 | `3` (기본) |
| FC 불확실 | `5` |
| FC 없이 영상만 | `0` |

---

### 커밋 `2213c23` — `cv2.putText` 한글 `?????` 표시 수정 (fix)

#### 원인

`cv2.putText()` + `FONT_HERSHEY_SIMPLEX`는 **ASCII(0~127)만 지원**합니다.  
한글(유니코드 0xAC00~0xD7A3)을 전달하면 렌더링 불가 → `?????` 로 표시됩니다.

#### 수정 위치 2곳

| 위치 | 수정 전 | 수정 후 |
|------|---------|---------|
| `KFModelCTRV.model_info` | `CTRV[직선]` | `CTRV[Line]` |
| `KFModelCTRV.model_info` | `CTRV[선회(15.3°/s)]` | `CTRV[Turn(15.3d/s)]` |
| ByteTrack 오버레이 | `활성:{n}` | `active:{n}` |

```python
# 수정 전
m = "직선" if abs(yr)<self._EPS else f"선회({math.degrees(yr):.1f}°/s)"

# 수정 후 (ASCII만 사용)
m = "Line" if abs(yr) < self._EPS else f"Turn({math.degrees(yr):.1f}d/s)"
```

---

### 커밋 `9265db9` — ArUco 마커 탐지기 통합 (feat)

#### ArUco 추가 배경

사용 중인 마커(50cm×50cm ArUco 마커)를 cv2.aruco로 직접 탐지하면 YOLO+깊이 방식보다 더 정밀한 3D 위치를 얻을 수 있습니다.

| 항목 | YOLO + RealSense depth | ArUco pose estimation |
|------|----------------------|----------------------|
| 중심 정밀도 | YOLO 박스 중심 (픽셀) | 4코너 평균 (서브픽셀) |
| 깊이 센서 | 필요 | **불필요** |
| 깊이 노이즈 영향 | 있음 | **없음** (기하학 계산) |
| 마커 ID 확인 | 불가 | **가능** |
| 6DOF 자세 | 부분적 | **완전** (roll/pitch/yaw) |

#### 추가된 CLI 인수 3개

```
--aruco on/off        ArUco 탐지 활성화 (기본: off)
--aruco-dict          사전 종류 (기본: 4X4_50)
                        4X4_50   : 빠름, 근거리
                        5X5_100  : 중간 거리
                        6X6_250  : 원거리, ID 최대 250
                        7X7_1000 : ID 최대 1000
--marker-size         마커 실물 크기 미터 (기본: 0.5m=50cm)
                      ※ 실제 마커와 반드시 일치해야 거리 추정 정확
```

#### 소스 우선순위 변경

```
[이전]
  1순위: YOLO + RealSense depth
  2순위: YOLO + FOV 각도 (depth 없을 때)

[현재]
  1순위: ArUco pose estimation  ← 신규 (깊이 센서 없이 tvec으로 3D 위치)
  2순위: YOLO + RealSense depth ← 기존
  3순위: YOLO + FOV 각도        ← 기존
```

#### ArUco 탐지 동작 원리

```python
# 1. 그레이스케일 변환 후 마커 탐지
gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
corners, ids, _ = aruco_detector.detectMarkers(gray)

# 2. 자세 추정 — tvec이 3D 오프셋(m) 직접 제공
rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners, MARKER_SIZE_M, camera_matrix, dist_coeffs)

ax = tvec[0][0]   # 가로 오프셋 (m)
ay = tvec[0][1]   # 세로 오프셋 (m)
az = tvec[0][2]   # 거리/깊이   (m)

# 3. 칼만 필터에 직접 입력 (YOLO 대체)
fx, fy, fz, vx, vy, vz, innov = single_kf.update(ax, ay, az)
```

#### 화면 표시 항목 (--aruco on 시)

```
┌────────────────────────────────────────────────────┐
│ SRC:ArUco|CTRV[Line]              (좌상단, 황색)   │
│                                                    │
│  ┌──────────────────┐                              │
│  │  ←── X축(빨강)   │  ← 3D 좌표축 표시           │
│  │  ↑ Y축(초록)     │                              │
│  │  Z축(파랑)→      │                              │
│  │  ArUco ID:0  Z:2.45m    (황색)                 │
│  └──────────────────┘                              │
│                                                    │
│  Offset X:0.12m  Y:-0.05m   (노란색)              │
│  Alt(Z): 2.45m              (초록색)              │
│  Vel Vx:0.03 Vy:-0.01 ...  (하늘색)               │
│  Speed:0.12m/s [12cm/s]     (파란색)              │
└────────────────────────────────────────────────────┘
```

#### OpenCV 버전 호환 처리

```python
# 탐지 API (4.7+ ArucoDetector / 구버전 detectMarkers 자동 선택)
try:
    _aruco_detector = cv2.aruco.ArucoDetector(adict, aparams)
    def _aruco_detect(gray): return _aruco_detector.detectMarkers(gray)
except AttributeError:
    def _aruco_detect(gray): return cv2.aruco.detectMarkers(gray, adict, ...)

# 축 그리기 API (4.8+ drawFrameAxes / 구버전 drawAxis 자동 폴백)
try:
    cv2.drawFrameAxes(img, cam_mat, dist, rvec, tvec, size)
except AttributeError:
    cv2.aruco.drawAxis(img, cam_mat, dist, rvec, tvec, size)
```

#### ByteTrack 모드 ArUco 보정

ByteTrack 모드에서는 주 트랙(highest score)의 중심과 ArUco 탐지 중심이 **150px 이내**일 때 추가 pose 보정을 적용합니다.

```
주 트랙 위치 (YOLO + depth)
    + ArUco pose (150px 이내)
    → 칼만 필터에 ArUco tvec 추가 보정
    → "ArUco refined ID:N" 오버레이 표시
```

---

## 3. 카테고리별 수정 항목 전체 목록

### 🔴 런타임 크래시 방지 (7건)

| 항목 | 내용 |
|------|------|
| `B1` | ByteTrack `_associate()` 빈 트랙/검출 시 반환값 오류 |
| `B2` | ByteTrack 3차 매칭 재진입 트랙 소멸 버그 |
| `C3` | CTRV EKF `np.linalg.inv(S)` 특이행렬 → `LinAlgError` |
| `C4` | KalmanBoxFilter h≈0 특이행렬 → `LinAlgError` |
| `C1` (d630197) | 필터 체인 각 단계 `as_depth_frame()` 미적용 → `AttributeError` |
| `M1` (d630197) | 벡터화 좌표 float 업캐스팅 → `get_distance()` `TypeError` |
| `C2`(원본) | `cudaMalloc` 반환코드 미확인 → NULL 포인터 segfault |

### 🟠 통신·안정성 (5건)

| 항목 | 내용 |
|------|------|
| `C2` | `send_mavlink()` 네트워크 오류(`OSError`) 미처리 → 메인루프 크래시 |
| `C7` | `cx,cy` 경계 밖 시 MAVLink 미송신 |
| `M4` | Heartbeat 타임아웃 시 `master=None` 미설정 → 의미없는 패킷 지속 |
| `edfba97` | `send_mavlink()` 송출 print 실수 삭제 후 복원 |
| `B5` | ByteTrack primary 0값 위치 혼동 출력 |

### 🟡 메모리·성능 (5건)

| 항목 | 내용 |
|------|------|
| `C5` | `MOTPEvaluator._log` 무제한 메모리 성장 |
| `C6` | LSTM 학습 메인루프 동기 블로킹 감지 |
| `M1` | Python 8400 순회 → numpy 벡터화 (5~15ms → 0.1ms) |
| `B4` | `track_color()` `np.random.seed()` 전역 상태 오염 |
| `B3` | `LSTMPredictor.predict()` `self.ref None` 안전장치 |

### 🟢 코드 품질 (4건)

| 항목 | 커밋 | 내용 |
|------|------|------|
| `C8` | `9baaeb8` | `if True:` 제거 후 블록 들여쓰기 재정렬 |
| `get_median_depth` | `9baaeb8` | `int()` 캐스팅 안전망 추가 |
| 한글→ASCII `2213c23` | `2213c23` | `CTRV[직선/선회]` → `CTRV[Line/Turn]`, `활성` → `active` |
| 소스 표시 | `9265db9` | `SRC:ArUco`, `SRC:YOLO+D` 구분 출력 |

### 🔵 성능·UX 개선 (7건)

| 항목 | 커밋 | 내용 |
|------|------|------|
| `P1` | `b0a37ba` | TensorRT 로딩 진행 메시지 + 소요시간 출력 |
| `P2` | `b0a37ba` | GPU 워밍업 더미 추론 (첫 프레임 JIT 지연 제거) |
| `P3` | `b0a37ba` | RealSense / GStreamer 초기화 진행 메시지 |
| `P4` | `b0a37ba` | MAVLink Heartbeat 타임아웃 5초 → 3초 단축 |
| `P5` | `b0a37ba` | 전체 초기화 총 소요시간 측정·출력 |
| `E1` | `e670f34` | `--mav` 인수 추가 (연결 주소 자유 설정) |
| `E2` | `e670f34` | `--mav-timeout` 인수 추가 (타임아웃 조절) |

### 🟣 ArUco 마커 탐지 (6건)

| 항목 | 커밋 | 내용 |
|------|------|------|
| `A1` | `9265db9` | `--aruco on/off` 인수 추가 |
| `A2` | `9265db9` | `--aruco-dict` 사전 선택 인수 추가 |
| `A3` | `9265db9` | `--marker-size` 실물 크기 인수 추가 (기본 0.5m) |
| `A4` | `9265db9` | ArUco 소스 우선순위 1순위 적용 (tvec 직접 3D) |
| `A5` | `9265db9` | 화면에 마커 ID·거리·3D 축 표시 |
| `A6` | `9265db9` | ByteTrack 모드 ArUco 150px 이내 보정 |

---

## 4. 원본 대비 주요 기능 추가 목록

```
원본 jetson_inference.py
    │
    ├── 칼만 필터 4모델 선택 (--model)
    │     ├── CV  : 등속도 6D 선형 (기본)
    │     ├── CA  : 등가속도 9D 선형
    │     ├── CTRV: 선회율 7D 비선형 EKF
    │     └── IMM : CV+CTRV 확률 혼합
    │
    ├── 추적 방식 선택 (--tracker)
    │     ├── single    : 최고 신뢰도 단일 객체 (원본과 동일 방식)
    │     └── bytetrack : 3단계 IoU 매칭 다중 객체 추적
    │                     (트랙별 독립 3D 칼만 + 고유 ID 부여)
    │
    ├── LSTM 미래 위치 예측 (--predict N)
    │     ├── PyTorch 온라인 학습 (2-layer LSTM)
    │     ├── 30프레임 과거 → N스텝 미래 예측
    │     └── PyTorch 미설치 시 선형 외삽 자동 대체
    │
    ├── MOTP 추적 정밀도 평가 (--motp)
    │     ├── Innovation Norm 기반 실시간 MOTP 계산
    │     └── 종료 시 motp_log.csv 자동 저장
    │
    ├── ArUco 마커 탐지 (--aruco on)
    │     ├── cv2.aruco 탐지기 (4X4_50 / 5X5_100 / 6X6_250 / 7X7_1000)
    │     ├── estimatePoseSingleMarkers → tvec 3D 위치 (깊이 센서 불필요)
    │     ├── 마커 ID·거리·3D 좌표축 화면 표시
    │     └── 소스 우선순위: ArUco > YOLO+depth > YOLO(FOV)
    │
    └── 하드웨어 처리 개선
          ├── RealSense 깊이 필터 체인 (spatial→temporal→hole_fill)
          ├── 5×5 중앙값 깊이 측정 (단일 픽셀 → 노이즈 강건)
          ├── intrinsics 1회 취득 (매 프레임 → 루프 밖)
          └── CUDA 메모리 자동 해제 (__del__ cudaFree)
```

---

## 5. 전체 CLI 인수 목록

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--stream` | `off` | GStreamer UDP 스트리밍 (`on` / `off`) |
| `--model` | `cv` | 칼만 필터 모델 (`cv` / `ca` / `ctrv` / `imm`) |
| `--tracker` | `single` | 추적 방식 (`single` / `bytetrack`) |
| `--predict` | `0` | LSTM 미래 예측 스텝 수 (0=비활성) |
| `--motp` | 비활성 | MOTP 추적 정밀도 평가 플래그 |
| `--mav` | `udpin:0.0.0.0:14551` | MAVLink 연결 주소 |
| `--mav-timeout` | `3` | Heartbeat 대기 타임아웃 (초) |
| `--aruco` | `off` | ArUco 마커 탐지 (`on` / `off`) |
| `--aruco-dict` | `4X4_50` | ArUco 사전 (`4X4_50` / `5X5_100` / `6X6_250` / `7X7_1000`) |
| `--marker-size` | `0.5` | 마커 실물 크기 미터 (현재 마커: 50cm×50cm) |

---

## 6. 환경별 실행 예시

```bash
# SITL 시뮬레이터 (PC에서 ArduPilot SITL 실행 중)
# → udpout으로 Jetson이 먼저 연결, Heartbeat 즉시 수신
python3 jetson_inference_advanced.py \
    --mav udpout:192.168.0.10:14550 \
    --mav-timeout 1 \
    --stream on --model ctrv

# 실기체 Pixhawk (USB-UART 직렬 연결) ← 가장 빠른 방식
python3 jetson_inference_advanced.py \
    --mav /dev/ttyUSB0 \
    --stream on --model ctrv

# 실기체 (특정 네트워크 인터페이스 고정)
python3 jetson_inference_advanced.py \
    --mav udpin:192.168.0.5:14551 \
    --mav-timeout 2 \
    --stream on

# 개발·테스트 (FC 없이 영상 처리만)
python3 jetson_inference_advanced.py \
    --mav udpin:0.0.0.0:14551 \
    --mav-timeout 0 \
    --stream off --model imm --tracker bytetrack

# 연구용 풀 옵션
python3 jetson_inference_advanced.py \
    --stream on --model ctrv --tracker bytetrack \
    --predict 15 --motp \
    --mav udpout:192.168.0.10:14550 --mav-timeout 1

# ArUco ON + CTRV 모델 (가장 정밀한 조합)
python3 jetson_inference_advanced.py \
    --aruco on --marker-size 0.5 \
    --model ctrv --stream on \
    --mav udpout:192.168.0.10:14550

# ArUco + ByteTrack 다중 마커 환경
python3 jetson_inference_advanced.py \
    --aruco on --aruco-dict 6X6_250 --marker-size 0.5 \
    --tracker bytetrack --model imm \
    --stream on --motp
```

---

## 7. 알려진 이슈 및 제약 사항

| 항목 | 내용 | 권장 조치 |
|------|------|-----------|
| 시작 시간 | TensorRT 로딩 5~20초 불가피 | 진행 메시지로 상태 확인 가능 |
| LSTM 학습 블로킹 | 50ms 이상 시 경고 출력 | `EPOCHS=3` 또는 `TRAIN_EVERY=30` 으로 줄이기 |
| ByteTrack `max_lost=30` | 소실 1초(30fps) 후 트랙 삭제 | 필요 시 `max_lost=60` 으로 늘리기 |
| MAVLink 재연결 없음 | 비행 중 연결 끊기면 복구 불가 | 추후 주기적 heartbeat 확인 로직 추가 필요 |
| CTRV Joseph P 업데이트 | 수치 정밀도 향상되나 연산량 증가 | 고부하 시 `--model cv` 사용 권장 |
| `--mav-timeout 0` | Heartbeat 없이 즉시 진행 | FC 없이 영상만 처리할 때만 사용 |
| ArUco `--marker-size` 불일치 | 실물 크기와 다르면 거리 오차 발생 | 실제 마커 크기(0.5m) 정확히 입력 필수 |
| ArUco + YOLO 동시 실행 | CPU 연산 약간 증가 | ArUco는 CPU 기반, GPU 부하 없음 |
| ArUco 탐지 실패 시 | 조명·각도에 따라 인식률 저하 | YOLO+depth 자동 폴백 |
| Windows IDE 모듈 오류 | cv2, pyrealsense2 등 Not Found 표시 | Jetson Nano 환경에서만 정상 실행됨 — Windows Python 환경 오탐 |

---

## 커밋 확인

```
git log --oneline drone_precision_landing/jetson_inference_advanced.py

9265db9 feat: ArUco 마커 탐지기 통합 (--aruco, --aruco-dict, --marker-size)
2213c23 fix: cv2.putText 한글 렌더링 불가 → ????? 표시 수정
e670f34 feat: MAVLink 연결 주소·타임아웃을 CLI 인수로 분리 (--mav, --mav-timeout)
b0a37ba perf: 시작 지연 원인 개선 — 진행 메시지·워밍업·타임아웃 단축
a810992 docs: jetson_inference_advanced.py 전체 수정 이력 매뉴얼 추가
d630197 fix: --stream on 실행 시 get_distance() 타입 오류 2건 수정
31c6ca5 fix: jetson_inference_modify.py 동일 결함 3건 advanced 버전에도 적용
edfba97 fix: C2 예외처리 추가 시 실수로 삭제된 MAVLink 송출 print 복원
9baaeb8 fix: 통신·메모리·수치 안정성 전방위 버그 8건 수정
a7e80b1 fix: ByteTrack 논리 버그 4건 및 안전성 개선 5건 수정
8a4c133 feat: drone_precision_landing 원본 기반 고도화 비전 시스템 추가
```

모든 변경사항은 `github.com/kwangsikjin-hue/precistion_landing` `main` 브랜치에 커밋되어 있습니다.

---

*본 매뉴얼은 `jetson_inference_advanced.py` 커밋 `9265db9` 기준으로 작성되었습니다.*
