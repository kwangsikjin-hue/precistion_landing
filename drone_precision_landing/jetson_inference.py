import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import ctypes
import math
import time
from pymavlink import mavutil

# --- [MAVLink 드론 비행 제어기 통신 초기화] ---
# USB-to-UART 모듈 연결 시 '/dev/ttyUSB0', 보드 핀 직접 연결 시 '/dev/ttyTHS1' 등으로 설정
try:
    master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=921600)
    master.wait_heartbeat()
    print("🛸 [성공] ArduPilot 비행 제어기(FC)와 MAVLink 연결이 완료되었습니다!")
except Exception as e:
    print(f"⚠️ FC 연결 실패 (통신 선로 또는 포트를 확인하세요): {e}")
    master = None

# --- [1. 메모리 오버헤드가 없는 Jetson 전용 정적 TensorRT 추론 클래스] ---
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

# --- [2. 하드웨어 세팅 (AI Engine 및 리얼센스 D435i)] ---
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
        output = output.reshape(5, 8400) # YOLOv8 바운딩 복원 폼
        predictions = output.T

        for pred in predictions:
            box = pred[:4]
            score = pred[4]
            
            if score > 0.6: # 오작동 방지를 위해 신뢰도 기준 60% 상향 검증
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
                        
                        # Jetson 터미널 로그 출력 (물리 cm 정밀도 모니터링)
                        print(f"🎯 패드 포착 -> 가로 오차: {offset_x*100:.1f}cm, 세로 오차: {offset_y*100:.1f}cm, 수직고도: {offset_z*100:.1f}cm")

                        # 4단계: 📡 [MAVLink] ArduPilot 전송 규격(삼각함수 라디안 각도)으로 포장하여 Pixhawk 송신
                        if master is not None:
                            angle_x = math.atan2(offset_x, offset_z)
                            angle_y = math.atan2(offset_y, offset_z)
                            
                            master.mav.landing_target_send(
                                int(time.time() * 1e6),             # 마이크로초 단위 타임스탬프
                                0,                                  # 타겟 번호 (0번 고정)
                                mavutil.mavlink.MAV_FRAME_BODY_NED, # 기체 이동 좌표계 매핑 정의
                                angle_x,                            # 가로 오프셋 각도 (Radian)
                                angle_y,                            # 세로 오프셋 각도 (Radian)
                                offset_z,                           # 수직 실제 고도 (Meter)
                                0, 0                                # 타겟 치수 정보 (기본값)
                            )

        # 최종 프레임 디스플레이 렌더링
        cv2.imshow('Jetson Edge Precision Landing Vision', color_image)
        
        # 키보드 'q'를 누르면 안전하게 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    # 하드웨어 스트림 채널 반환 및 윈도우 파괴
    pipeline.stop()
    cv2.destroyAllWindows()
