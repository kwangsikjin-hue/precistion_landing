import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import ctypes
import math
import time
from pymavlink import mavutil

# --- GStreamer 스트리밍 파이프라인 ---
# [B6 수정] ! 구분자 양쪽에 공백 추가 (일부 GStreamer 버전 파싱 오류 방지)
gst_pipeline = (
    "appsrc ! videoconvert ! "
    "video/x-raw,format=I420 ! "
    "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
    "rtph264pay ! "
    "udpsink host=192.168.1.30 port=5600"
)
out = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))

# [B4 보완] VideoWriter 초기화 실패 시 조기 경고
if not out.isOpened():
    print("⚠️ GStreamer VideoWriter 초기화 실패 — 스트리밍 없이 계속 진행합니다.")


# --- MAVLink 비행 제어기 연결 ---
try:
    master = mavutil.mavlink_connection('udp:127.0.0.1:14551')
    master.wait_heartbeat()
    print("🛸 [성공] ArduPilot FC와 MAVLink 연결 완료!")
except Exception as e:
    print(f"⚠️ FC 연결 실패: {e}")
    master = None

# MAVLink 송신 주기 제어 (30fps 전부 송신 시 FC 버스 과부하 방지)
MAVLINK_SEND_HZ = 10
_last_mav_send = 0.0


# --- [1] TensorRT 추론 엔진 ---
class JetsonTRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # [B9] cudaMalloc 포인터 목록 저장 → __del__ 에서 해제
        self.cuda_lib = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self._cuda_ptrs = []  # 해제를 위해 별도 보관

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            if shape[0] == -1 or shape[0] == 0:
                shape = (1, 3, 640, 640) if self.engine.binding_is_input(binding) else (1, 5, 8400)

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = np.empty(shape, dtype=dtype)

            cuda_ptr = ctypes.c_void_p()
            self.cuda_lib.cudaMalloc(ctypes.byref(cuda_ptr), host_mem.nbytes)
            self._cuda_ptrs.append(cuda_ptr)

            self.bindings.append(int(cuda_ptr.value))
            buf = {'host': host_mem, 'device': cuda_ptr.value, 'bytes': host_mem.nbytes}
            (self.inputs if self.engine.binding_is_input(binding) else self.outputs).append(buf)

    # [B9 수정] 소멸자에서 CUDA 메모리 해제
    def __del__(self):
        for ptr in self._cuda_ptrs:
            if ptr.value:
                self.cuda_lib.cudaFree(ptr)

    def infer(self, img):
        img_resized = cv2.resize(img, (640, 640))
        img_in = np.ascontiguousarray(
            img_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
        )
        img_in = np.expand_dims(img_in, axis=0)

        np.copyto(self.inputs[0]['host'], img_in)
        self.cuda_lib.cudaMemcpy(
            ctypes.c_void_p(self.inputs[0]['device']),
            self.inputs[0]['host'].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.inputs[0]['bytes']),
            ctypes.c_int(1)   # cudaMemcpyHostToDevice
        )
        self.context.execute_v2(bindings=self.bindings)
        self.cuda_lib.cudaMemcpy(
            self.outputs[0]['host'].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self.outputs[0]['device']),
            ctypes.c_size_t(self.outputs[0]['bytes']),
            ctypes.c_int(2)   # cudaMemcpyDeviceToHost
        )
        return self.outputs[0]['host']


# --- [2] 칼만 필터 (3D 위치 + 속도 동시 추정) ---
#
# 상태 벡터 (6D): [x, y, z, vx, vy, vz]
# 측정 벡터 (3D): [x, y, z]   ← RealSense 실측
#
# 등속도 모델:
#   pos(t+dt) = pos(t) + vel(t)*dt
#   vel(t+dt) = vel(t)
#
# 노이즈 행렬 의미:
#   Q (processNoiseCov)      : 모델 불확실성 — 클수록 측정값을 더 빠르게 추종
#   R (measurementNoiseCov)  : 센서 잡음     — RealSense D435i 깊이 정밀도 반영
class KalmanFilter3D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)

        # 측정 행렬 H: 상태→측정 (위치만 관측)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # 전이 행렬 F: dt 항은 매 프레임 갱신
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)

        # 시스템 잡음 Q
        q = np.zeros((6, 6), dtype=np.float32)
        q[0:3, 0:3] = np.eye(3) * 1e-3   # 위치 프로세스 잡음
        q[3:6, 3:6] = np.eye(3) * 1e-1   # 속도 프로세스 잡음
        self.kf.processNoiseCov = q

        # 측정 잡음 R: RealSense D435i 깊이 정밀도 ~1cm
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

        self.initialized = False
        self.prev_time = None
        self._reset_cov()

    def _reset_cov(self):
        # [B2/B3 수정] 초기화·재추적 시 오차 공분산 P 완전 리셋
        p = np.eye(6, dtype=np.float32)
        p[0:3, 0:3] *= 1.0    # 위치 초기 불확실성
        p[3:6, 3:6] *= 10.0   # 속도 초기 불확실성 (처음엔 속도를 모름)
        self.kf.errorCovPost = p

    def _init_state(self, x, y, z, t):
        self._reset_cov()
        self.kf.statePost = np.array(
            [[x], [y], [z], [0.0], [0.0], [0.0]], dtype=np.float32
        )
        self.initialized = True
        self.prev_time = t

    def update(self, x, y, z):
        """
        RealSense 3D 측정값 → 칼만 보정 위치 + 속도 반환
        반환: (fx, fy, fz, vx, vy, vz)
        """
        current_time = time.time()

        if not self.initialized:
            self._init_state(x, y, z, current_time)
            return x, y, z, 0.0, 0.0, 0.0

        dt = max(current_time - self.prev_time, 1e-3)
        self.prev_time = current_time

        # 전이 행렬 F에 실제 dt 반영
        self.kf.transitionMatrix[0, 3] = dt
        self.kf.transitionMatrix[1, 4] = dt
        self.kf.transitionMatrix[2, 5] = dt

        self.kf.predict()
        corrected = self.kf.correct(
            np.array([[x], [y], [z]], dtype=np.float32)
        )

        return (
            float(corrected[0]), float(corrected[1]), float(corrected[2]),
            float(corrected[3]), float(corrected[4]), float(corrected[5]),
        )

    def reset(self):
        self.initialized = False
        self.prev_time = None
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
        self._reset_cov()  # [B2 수정] 오차 공분산도 함께 리셋


# --- [3] 하드웨어 초기화 ---
trt_brain = JetsonTRTEngine('best.engine')

pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(rs_config)

align = rs.align(rs.stream.color)

# RealSense 깊이 후처리 필터 체인 (성능 향상)
#   spatial  : 인접 픽셀 공간 평균 → 깊이 홀 감소
#   temporal : 이전 프레임 혼합   → 시간 축 노이즈 감소
#   hole_fill: 남은 홀 채우기
spatial_filter  = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_fill       = rs.hole_filling_filter()

# [B1 수정] intrinsics는 세션 중 변하지 않으므로 루프 밖에서 1회만 취득
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 유효 깊이 범위 (m) — 배경 오인식 방지 [B8 수정]
DEPTH_MIN_M = 0.15   # RealSense D435i 최소 신뢰 거리
DEPTH_MAX_M = 8.0    # 정밀 착륙 유효 고도 상한

print("🚀 [성공] AI 추론 엔진 및 리얼센스 파이프라인 가동 완료.")

# --- [4] 칼만 필터 인스턴스 ---
kf_tracker = KalmanFilter3D()


def get_median_depth(depth_frame, cx, cy, half=2):
    """
    [B7 수정] 단일 픽셀 대신 (2*half+1)² 영역의 유효 깊이 중앙값 반환.
    깊이 홀(0값) 픽셀은 제외하고 계산한다.
    """
    vals = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 640 and 0 <= ny < 480:
                d = depth_frame.get_distance(nx, ny)
                if d > 0:
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 깊이 후처리 필터 적용
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_fill.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())

        # TensorRT 추론
        output = trt_brain.infer(color_image)
        output = output.reshape(5, 8400)
        predictions = output.T

        # 신뢰도 최고 단일 객체 선택
        best_pred  = None
        best_score = 0.6

        for pred in predictions:
            score = pred[4]
            if score > best_score:
                best_score = score
                best_pred  = pred

        if best_pred is not None:
            x_center, y_center, w, h = best_pred[:4]

            x1 = int(x_center - w / 2)
            y1 = int((y_center - h / 2) * (480 / 640))
            x2 = int(x_center + w / 2)
            y2 = int((y_center + h / 2) * (480 / 640))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if 0 <= cx < 640 and 0 <= cy < 480:
                # [B7 수정] 5×5 영역 중앙값 깊이
                depth_value = get_median_depth(depth_frame, cx, cy, half=2)

                # [B8 수정] 유효 범위 체크 추가
                if DEPTH_MIN_M < depth_value < DEPTH_MAX_M:
                    raw_x, raw_y, raw_z = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [cx, cy], depth_value
                    )

                    # 칼만 필터 보정
                    fx, fy, fz, vx, vy, vz = kf_tracker.update(raw_x, raw_y, raw_z)
                    speed = math.sqrt(vx**2 + vy**2 + vz**2)

                    # 화면 시각화
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

                    cv2.putText(color_image,
                                f"Offset X:{fx:.2f}m  Y:{fy:.2f}m",
                                (x1, y1 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
                    cv2.putText(color_image,
                                f"Alt(Z): {fz:.2f}m",
                                (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    cv2.putText(color_image,
                                f"Vel Vx:{vx:.2f} Vy:{vy:.2f} Vz:{vz:.2f} m/s",
                                (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
                    cv2.putText(color_image,
                                f"Speed: {speed:.2f} m/s  [{speed*100:.1f} cm/s]",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 2)

                    print(
                        f"🎯 위치(보정) X:{fx*100:.1f}cm Y:{fy*100:.1f}cm Z:{fz*100:.1f}cm | "
                        f"🚀 속도 Vx:{vx*100:.1f} Vy:{vy*100:.1f} Vz:{vz*100:.1f} cm/s | "
                        f"합성:{speed*100:.1f} cm/s"
                    )

                    # MAVLink 송신 — 10Hz 제한 (FC 버스 부하 감소)
                    now = time.time()
                    if master is not None and (now - _last_mav_send) >= 1.0 / MAVLINK_SEND_HZ:
                        _last_mav_send = now
                        angle_x = math.atan2(fx, fz)
                        angle_y = math.atan2(fy, fz)
                        master.mav.landing_target_send(
                            int(now * 1e6),
                            0,
                            mavutil.mavlink.MAV_FRAME_BODY_NED,
                            angle_x,
                            angle_y,
                            fz,
                            0, 0
                        )
        else:
            kf_tracker.reset()

        if out.isOpened():
            out.write(color_image)
        cv2.imshow('Jetson Precision Landing [Kalman+Velocity]', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    out.release()          # [B4 수정] GStreamer 스트림 정상 종료
    cv2.destroyAllWindows()
