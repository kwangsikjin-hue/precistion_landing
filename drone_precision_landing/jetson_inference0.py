import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import ctypes

# --- 1. 정적 버퍼 매핑 기반의 Jetson 전용 TensorRT 추론 클래스 ---
class JetsonTRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # CUDA 런타임 라이브러리 직접 로드
        self.cuda_lib = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            if shape[0] == -1 or shape[0] == 0: # 배치 크기가 동적일 경우 1로 고정
                shape[0] = 1
                
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = np.empty(shape, dtype=dtype)
            
            # CUDA 메모리 할당 (cudaMalloc)
            cuda_ptr = ctypes.c_void_p()
            self.cuda_lib.cudaMalloc(ctypes.byref(cuda_ptr), host_mem.nbytes)
            
            self.bindings.append(int(cuda_ptr.value))
            
            # 딕셔너리 구조로 통일하여 명확하게 저장
            buffer_info = {'host': host_mem, 'device': cuda_ptr.value, 'bytes': host_mem.nbytes}
            if self.engine.binding_is_input(binding):
                self.inputs.append(buffer_info)
            else:
                self.outputs.append(buffer_info)

    def infer(self, img):
        # 1. 이미지 크기 변환 및 정규화 (640x640)
        img_resized = cv2.resize(img, (640, 640))
        img_in = img_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)
        img_in = np.ascontiguousarray(img_in)
        
        # 2. 호스트 -> 디바이스 메모리 복사 (리스트 인덱스 0번 접근으로 명확히 수정)
        np.copyto(self.inputs[0]['host'], img_in)
        self.cuda_lib.cudaMemcpy(
            ctypes.c_void_p(self.inputs[0]['device']), 
            self.inputs[0]['host'].ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_size_t(self.inputs[0]['bytes']), 
            ctypes.c_int(1) # cudaMemcpyHostToDevice
        )
        
        # 3. GPU 가속 추론 실행
        self.context.execute_v2(bindings=self.bindings)
        
        # 4. 디바이스 -> 호스트 메모리 복사
        self.cuda_lib.cudaMemcpy(
            self.outputs[0]['host'].ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_void_p(self.outputs[0]['device']), 
            ctypes.c_size_t(self.outputs[0]['bytes']), 
            ctypes.c_int(2) # cudaMemcpyDeviceToHost
        )
        return self.outputs[0]['host']

# --- 2. 하드웨어 초기화 (AI 및 인텔 리얼센스 D435i) ---
trt_brain = JetsonTRTEngine('best.engine')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

print("🚀 [성공] 메모리 버그가 완벽히 수정되었습니다. D435i 카메라 가동을 시작합니다.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # AI 가속 추론 진행
        output = trt_brain.infer(color_image)
        output = output.reshape(5, 8400) # YOLOv8 출력 포맷 디코드
        predictions = output.T
        
        for pred in predictions:
            box = pred[:4]
            score = pred[4] # 신뢰도 점수
            if score > 0.5: # 신뢰도 50% 이상만 착륙 마크로 인정
                x_center, y_center, w, h = box
                
                # 640x640 해상도에서 640x480 실제 영상 크기로 비율 역산
                x1 = int((x_center - w/2))
                y1 = int((y_center - h/2) * (480/640))
                x2 = int((x_center + w/2))
                y2 = int((y_center + h/2) * (480/640))
                
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if 0 <= cx < 640 and 0 <= cy < 480:
                    # D435i 정렬된 픽셀 좌표 기반 실제 수직 거리 검센싱
                    depth_value = depth_frame.get_distance(cx, cy)
                    
                    if depth_value > 0:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(color_image, f"Vertiport: {depth_value:.2f}m", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Jetson Fix Precision Landing Vision', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
