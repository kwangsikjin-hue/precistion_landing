import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import ctypes
import math
import time
import argparse  # 명령줄 인자를 받기 위한 모듈 추가
from pymavlink import mavutil
import subprocess

# --- [0. 명령줄 인자(Argument) 파싱 설정] ---
parser = argparse.ArgumentParser(description="Jetson Nano AI Inference and GStreamer Streaming")
parser.add_argument('--stream', type=str, default='off', choices=['on', 'off'],
                    help="GStreamer UDP 스트리밍을 켤지 끌지 결정합니다. ('on' 또는 'off')")
args = parser.parse_args()

# 스트리밍 활성화 여부를 변수에 저장
ENABLE_STREAMING = (args.stream == 'on')

# --- [1. GStreamer 파이프라인 초기화 (스트리밍 ON일 때만 실행)] ---
if ENABLE_STREAMING:
    print("📡 [알림] GStreamer UDP 스트리밍 모드가 켜졌습니다. (Target IP: 192.168.0.30:15600)")
    gst_pipeline = (
        "appsrc ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        # bitrate를 1500에서 800으로 낮춰 네트워크 부하를 줄이고, key-int-max를 15로 줄여 화면 복구 속도 향상
        "x264enc tune=zerolatency bitrate=800 speed-preset=superfast key-int-max=15 ! "
        "h264parse ! "
        # config-interval을 1로 유지하여 메타데이터를 자주 보냄
        "rtph264pay pt=96 config-interval=1 ! "
        # 버퍼링 안정성을 위해 큐(queue) 추가
        "queue max-size-buffers=10 leaky=downstream ! "
        "udpsink host=192.168.0.30 port=15600 sync=false async=false"
    )
    out = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))
else:
    print("🖥️ [알림] GStreamer 스트리밍이 꺼져 있습니다. (Local 화면만 표시)")
    out = None

# --- [MAVLink 드론 비행 제어기 통신 초기화] ---
try:
    print("⏳ MAVLink 연결 시도 중...")
    # [범용적 솔루션] udpin:0.0.0.0:14551을 사용하여 에러 99를 방지하고,
    # SITL PC(또는 로컬호스트)로부터 오는 패킷을 모든 네트워크 카드에서 수신 대기합니다.
    master = mavutil.mavlink_connection('udpin:0.0.0.0:14551')
    
    # 무한 대기(프리징) 방지를 위해 타임아웃 5초 설정
    msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
    if msg:
        print("🛸 [성공] ArduPilot 비행 제어기(FC)와 MAVLink 연결 완료!")
    else:
        print("⚠️ [경고] 5초간 MAVLink Heartbeat가 없습니다. 통신 없이 영상 처리를 강행합니다.")
except Exception as e:
    print(f"⚠️ FC 연결 실패: {e}")
    master = None

# --- [2. 메모리 오버헤드가 없는 Jetson 전용 정적 TensorRT 추론 클래스] ---
class JetsonTRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Jetson 전용 CUDA 런타임 라이브러리 직접 로드
        self.cuda_lib = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            # 동적 배치 형태 우회 처리
            if shape[0] == -1 or shape[0] == 0:
                shape = (1, 3, 640, 640) if self.engine.binding_is_input(binding) else (1, 5, 8400)

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = np.empty(shape, dtype=dtype)

            # CUDA 정적 메모리 할당 (cudaMalloc)
            cuda_ptr = ctypes.c_void_p()
            self.cuda_lib.cudaMalloc(ctypes.byref(cuda_ptr), host_mem.nbytes)
            self.bindings.append(int(cuda_ptr.value))

            buffer_info = {'host': host_mem, 'device': cuda_ptr.value, 'bytes': host_mem.nbytes}
            if self.engine.binding_is_input(binding):
                self.inputs.append(buffer_info)
            else:
                self.outputs.append(buffer_info)

    def infer(self, img):
        # 이미지 전처리 및 640x640 정규화
        img_resized = cv2.resize(img, (640, 640))
        img_in = img_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)
        img_in = np.ascontiguousarray(img_in)

        # 호스트 -> 디바이스 메모리 복사
        np.copyto(self.inputs[0]['host'], img_in)
        self.cuda_lib.cudaMemcpy(
            ctypes.c_void_p(self.inputs[0]['device']),
            self.inputs[0]['host'].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.inputs[0]['bytes']),
            ctypes.c_int(1)
        )

        # GPU 가속 핵심 추론 실행
        self.context.execute_v2(bindings=self.bindings)

        # 디바이스 -> 호스트 메모리 복사
        self.cuda_lib.cudaMemcpy(
            self.outputs[0]['host'].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self.outputs[0]['device']),
            ctypes.c_size_t(self.outputs[0]['bytes']),
            ctypes.c_int(2)
        )
        return self.outputs[0]['host']

# --- [3. 하드웨어 세팅 (AI Engine 및 리얼센스 D435i)] ---
trt_brain = JetsonTRTEngine('best.engine')

pipeline = rs.pipeline()
config = rs.config()

# USB 대역폭 및 젯슨 리소스 안정화를 고려한 정밀 해상도 맵 세팅
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 2D 프레임 좌표와 3D 물리 공간의 중심을 정렬하는 오프셋 얼라인 객체
align = rs.align(rs.stream.color)

print("🚀 [성공] AI 추론 엔진 및 리얼센스 정렬 파이프라인 가동 완료.")

try:
    while True:
        # 센서로부터 실시간 비전 신호 대기
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # 리얼센스 고유 렌즈 내상수(Intrinsics) 행렬 실시간 로드
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # TensorRT AI 가속 추론 진행
        output = trt_brain.infer(color_image)
        output = output.reshape(5, 8400)  # YOLOv8 바운딩 복원 폼
        predictions = output.T

        for pred in predictions:
            box = pred[:4]
            score = pred[4]

            if score > 0.6:  # 오작동 방지를 위해 신뢰도 기준 60% 상향 검증
                x_center, y_center, w, h = box

                # 640x640 추론 해상도에서 640x480 실제 픽셀 평면으로 비율 복원
                x1 = int((x_center - w/2))
                y1 = int((y_center - h/2) * (480/640))
                x2 = int((x_center + w/2))
                y2 = int((y_center + h/2) * (480/640))

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if 0 <= cx < 640 and 0 <= cy < 480:
                    # 1단계: 마크 정중앙의 수직 깊이(Z고도) 거리 산출
                    depth_value = depth_frame.get_distance(cx, cy)

                    if depth_value > 0:
                        # 2단계: 🔥 [기하학 매핑] 2D 픽셀을 물리 미터(m) 단위의 3D 상대 좌표로 완벽 역산 변환
                        camera_3d_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_value)

                        offset_x = camera_3d_point[0]  # 가로 오프셋 오차 (m)
                        offset_y = camera_3d_point[1]  # 세로 오프셋 오차 (m)
                        offset_z = camera_3d_point[2]  # 실제 수직 고도 (Z축 m)

                        # 3단계: 모니터 화면 시각화 출력 (가로/세로 오프셋 실시간 박스 표기)
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

                        text_dist = f"Offset X: {offset_x:.2f}m, Y: {offset_y:.2f}m"
                        text_alt = f"Alt(Z): {offset_z:.2f}m"
                        cv2.putText(color_image, text_dist, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(color_image, text_alt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Jetson 터미널 로그 출력
                        print(f"🎯 패드 포착 -> 가로 오차: {offset_x*100:.1f}cm, 세로 오차: {offset_y*100:.1f}cm, 수직고도: {offset_z*100:.1f}cm")

                        # 4단계: FOV 기반 정밀 착륙 방사각(angle_x, angle_y) 계산
                        dx = cx - intrinsics.ppx
                        dy = cy - intrinsics.ppy
                        angle_x = math.atan2(dx, intrinsics.fx)
                        angle_y = math.atan2(dy, intrinsics.fy)

                        # 5단계: MAVLink LANDING_TARGET 메시지 송출 (10~30Hz 주기)
                        if master is not None:
                            try:
                                tnow = int(time.time() * 1e6)
                                frame = getattr(mavutil.mavlink, 'MAV_FRAME_BODY_NED', 8)
                                try:
                                    # [방법 A] MAVLink 2 옵션 인자가 지원되는 최신 pymavlink 구문 시도
                                    master.mav.landing_target_send(
                                        time_usec=tnow,
                                        target_num=0,
                                        frame=frame,
                                        angle_x=angle_x,
                                        angle_y=angle_y,
                                        distance=depth_value,
                                        size_x=0.0,
                                        size_y=0.0,
                                        type=2,  # LANDING_TARGET_TYPE_VISION_FIDUCIAL
                                        position_valid=0
                                    )
                                except TypeError:
                                    # [방법 B] 구형 pymavlink (MAVLink 1 호환)인 경우 8개 기본 위치 인자만 전송
                                    master.mav.landing_target_send(
                                        tnow,         # time_usec
                                        0,            # target_num
                                        frame,        # frame
                                        angle_x,      # angle_x
                                        angle_y,      # angle_y
                                        depth_value,  # distance
                                        0.0,          # size_x
                                        0.0           # size_y
                                    )
                                print(f"📡 [MAVLink] LANDING_TARGET 송출 -> angle_x: {math.degrees(angle_x):.2f}°, angle_y: {math.degrees(angle_y):.2f}°, Dist: {depth_value:.2f}m")
                            except Exception as mav_err:
                                print(f"⚠️ MAVLink LANDING_TARGET 송신 실패: {mav_err}")

        # --- [프레임 송신 및 로컬 화면 출력] ---
        
        # 1. Mission Planner (PC)로 GStreamer 영상 송신 (명령줄 옵션이 on일 때만 작동)
        if ENABLE_STREAMING and out is not None and out.isOpened():
            out.write(color_image)
            
        # 2. 젯슨 나노 로컬 모니터에 영상 출력
        cv2.imshow("Jetson Local View", color_image)
        
        # 'q' 키를 누르면 루프 탈출 및 프로그램 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- [프로그램 안전 종료 및 자원 해제] ---
except KeyboardInterrupt:
    print("사용자에 의해 프로그램을 종료합니다.")
except Exception as e:
    print(f"메인 루프 실행 중 오류 발생: {e}")
finally:
    print("시스템 자원을 해제합니다...")
    pipeline.stop()
    cv2.destroyAllWindows()
    if ENABLE_STREAMING and out is not None and out.isOpened():
        out.release()