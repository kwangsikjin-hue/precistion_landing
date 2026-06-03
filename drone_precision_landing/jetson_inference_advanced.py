#!/usr/bin/env python3
"""
jetson_inference_advanced.py — Jetson 정밀 착륙 비전 시스템 고도화
원본: drone_precision_landing/jetson_inference.py 기반

실행 예:
  python3 jetson_inference_advanced.py --stream on --model ctrv --tracker bytetrack --predict 15 --motp
  python3 jetson_inference_advanced.py --stream off --model imm --tracker bytetrack
  python3 jetson_inference_advanced.py --stream on  --model ca  --motp

인수:
  --stream  {on,off}              GStreamer UDP 스트리밍 (기본: off)
  --model   {cv,ca,ctrv,imm}     칼만 필터 모델 (기본: cv)
  --tracker {single,bytetrack}   추적 방식 (기본: single)
  --predict N                    LSTM 미래 N 스텝 예측 (0=비활성)
  --motp                         MOTP 추적 정밀도 평가 출력
"""

import argparse
import csv
import math
import time
import ctypes
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
from pymavlink import mavutil

# PyTorch — LSTM 예측기 선택적 의존성
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# scipy — ByteTrack 헝가리안 매칭 (없으면 그리디 폴백)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────
# 커맨드라인 인수
# ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Jetson Nano AI Inference — 고도화 버전")
parser.add_argument('--stream', type=str, default='off', choices=['on', 'off'],
                    help="GStreamer UDP 스트리밍 켜기/끄기 (기본: off)")
parser.add_argument('--model', choices=['cv', 'ca', 'ctrv', 'imm'], default='cv',
                    help='칼만 필터 모델 (기본: cv)\n'
                         '  cv   : 등속도 6D 선형\n'
                         '  ca   : 등가속도 9D 선형\n'
                         '  ctrv : 선회율 7D EKF (비선형)\n'
                         '  imm  : CV+CTRV 확률 혼합')
parser.add_argument('--tracker', choices=['single', 'bytetrack'], default='single',
                    help='추적 방식 (기본: single)\n'
                         '  single    : 최고 신뢰도 단일 객체\n'
                         '  bytetrack : 다중 객체 ByteTrack (ID 유지)')
parser.add_argument('--predict', type=int, default=0, metavar='N',
                    help='LSTM 미래 위치 예측 스텝 수 (0=비활성)')
parser.add_argument('--motp', action='store_true',
                    help='MOTP 추적 정밀도 평가 활성화')
parser.add_argument('--mav', type=str, default='udpin:0.0.0.0:14551',
                    help='MAVLink 연결 주소 (기본: udpin:0.0.0.0:14551)\n'
                         '  예) SITL PC:       --mav udpout:192.168.0.10:14550\n'
                         '      특정 인터페이스: --mav udpin:192.168.0.5:14551\n'
                         '      USB-UART:       --mav /dev/ttyUSB0\n'
                         '  ※ udpout은 Jetson이 FC에 먼저 연결 → Heartbeat 빠름')
parser.add_argument('--mav-timeout', type=int, default=3, metavar='SEC',
                    help='MAVLink Heartbeat 대기 타임아웃 초 (기본: 3)')
args = parser.parse_args()

ENABLE_STREAMING = (args.stream == 'on')
print(f"[설정] 스트리밍={args.stream} | 모델={args.model.upper()} | "
      f"추적={args.tracker} | LSTM={args.predict}스텝 | MOTP={args.motp} | "
      f"MAVLink={args.mav}")

# ─────────────────────────────────────────────────────────────────────
# GStreamer 스트리밍 초기화
# ─────────────────────────────────────────────────────────────────────
if ENABLE_STREAMING:
    print("📡 [알림] GStreamer UDP 스트리밍 초기화 중... (x264enc 로딩, 2~5초 소요)")
    gst_pipeline = (
        "appsrc ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency bitrate=800 speed-preset=superfast key-int-max=15 ! "
        "h264parse ! "
        "rtph264pay pt=96 config-interval=1 ! "
        "queue max-size-buffers=10 leaky=downstream ! "
        "udpsink host=192.168.0.30 port=15600 sync=false async=false"
    )
    out = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))
    if not out.isOpened():
        print("⚠️ GStreamer 초기화 실패 — 스트리밍 없이 계속합니다.")
        out = None
    else:
        print("✅ GStreamer 스트리밍 초기화 완료 (Target: 192.168.0.30:15600)")
else:
    print("🖥️  [알림] GStreamer 스트리밍 OFF (로컬 화면만 표시)")
    out = None

# ─────────────────────────────────────────────────────────────────────
# MAVLink 비행 제어기 연결
# ─────────────────────────────────────────────────────────────────────
try:
    print(f"⏳ MAVLink 연결 시도 중... [{args.mav}] (최대 {args.mav_timeout}초 대기)")
    master = mavutil.mavlink_connection(args.mav)
    msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=args.mav_timeout)
    if msg:
        print(f"🛸 [성공] ArduPilot FC MAVLink 연결 완료! [{args.mav}]")
    else:
        # [M4] 타임아웃 시 master=None
        print(f"⚠️ [경고] {args.mav_timeout}초간 Heartbeat 없음 — master=None 처리 후 영상 처리만 진행합니다.")
        master = None
except Exception as e:
    print(f"⚠️ FC 연결 실패 [{args.mav}]: {e}")
    master = None

MAVLINK_SEND_HZ = 10
_last_mav_send  = 0.0


# ─────────────────────────────────────────────────────────────────────
# [1] TensorRT 추론 엔진
# ─────────────────────────────────────────────────────────────────────
class JetsonTRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        # TensorRT 역직렬화 — Jetson Nano에서 5~20초 소요, 진행 메시지 표시
        print(f"⏳ TensorRT 엔진 로딩 중: {engine_path} (5~20초 소요, 잠시 대기)")
        t0 = time.time()
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context  = self.engine.create_execution_context()
        print(f"✅ TensorRT 엔진 로드 완료 ({time.time()-t0:.1f}초)")
        self.cuda_lib = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        self.inputs, self.outputs, self.bindings, self._ptrs = [], [], [], []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            if shape[0] in (-1, 0):
                shape = (1,3,640,640) if self.engine.binding_is_input(binding) else (1,5,8400)
            dtype    = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = np.empty(shape, dtype=dtype)
            cuda_ptr = ctypes.c_void_p()
            # [C2] cudaMalloc 반환 코드 확인 — 0이 아니면 GPU 메모리 부족
            ret = self.cuda_lib.cudaMalloc(ctypes.byref(cuda_ptr), host_mem.nbytes)
            if ret != 0:
                raise RuntimeError(
                    f"cudaMalloc 실패 (에러코드: {ret}) — GPU 메모리 부족 가능성")
            self._ptrs.append(cuda_ptr)
            self.bindings.append(int(cuda_ptr.value))
            buf = {'host': host_mem, 'device': cuda_ptr.value, 'bytes': host_mem.nbytes}
            (self.inputs if self.engine.binding_is_input(binding) else self.outputs).append(buf)

    def __del__(self):
        for p in self._ptrs:
            if p.value:
                self.cuda_lib.cudaFree(p)

    def infer(self, img):
        img_in = np.ascontiguousarray(
            cv2.resize(img,(640,640)).transpose(2,0,1).astype(np.float32)/255.0
        )[np.newaxis]
        np.copyto(self.inputs[0]['host'], img_in)
        self.cuda_lib.cudaMemcpy(
            ctypes.c_void_p(self.inputs[0]['device']),
            self.inputs[0]['host'].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.inputs[0]['bytes']), ctypes.c_int(1))
        self.context.execute_v2(bindings=self.bindings)
        self.cuda_lib.cudaMemcpy(
            self.outputs[0]['host'].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self.outputs[0]['device']),
            ctypes.c_size_t(self.outputs[0]['bytes']), ctypes.c_int(2))
        return self.outputs[0]['host']


# ─────────────────────────────────────────────────────────────────────
# [2] 칼만 필터 모델 — CV / CA / CTRV / IMM
# ─────────────────────────────────────────────────────────────────────
class KFModelCV:
    """등속도 6D 선형 칼만: 상태 [x,y,z, vx,vy,vz]"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]], dtype=np.float32)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        q = np.zeros((6,6), dtype=np.float32)
        q[0:3,0:3]=np.eye(3)*1e-3; q[3:6,3:6]=np.eye(3)*1e-1
        self.kf.processNoiseCov     = q
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32)*1e-2
        self.initialized=False; self.prev_time=None; self._reset_cov()

    def _reset_cov(self):
        p=np.eye(6,dtype=np.float32); p[3:6,3:6]*=10.0
        self.kf.errorCovPost=p

    def _init(self, x, y, z, t):
        self._reset_cov()
        self.kf.statePost=np.array([[x],[y],[z],[0],[0],[0]],dtype=np.float32)
        self.initialized=True; self.prev_time=t

    def update(self, x, y, z):
        now=time.time()
        if not self.initialized:
            self._init(x,y,z,now); return x,y,z,0.,0.,0.,0.
        dt=max(now-self.prev_time,1e-3); self.prev_time=now
        self.kf.transitionMatrix[0,3]=self.kf.transitionMatrix[1,4]=\
            self.kf.transitionMatrix[2,5]=dt
        pred=self.kf.predict()
        meas=np.array([[x],[y],[z]],dtype=np.float32)
        innov=float(np.linalg.norm(meas.flatten()-pred[:3].flatten()))
        corr=self.kf.correct(meas)
        return float(corr[0]),float(corr[1]),float(corr[2]),\
               float(corr[3]),float(corr[4]),float(corr[5]),innov

    def reset(self):
        self.initialized=False; self.prev_time=None
        self.kf.statePost=np.zeros((6,1),dtype=np.float32); self._reset_cov()

    @property
    def model_info(self): return "CV"


class KFModelCA:
    """등가속도 9D 선형 칼만: 상태 [x,y,z, vx,vy,vz, ax,ay,az]"""
    def __init__(self):
        self.kf=cv2.KalmanFilter(9,3)
        H=np.zeros((3,9),dtype=np.float32); H[0,0]=H[1,1]=H[2,2]=1.0
        self.kf.measurementMatrix=H
        self.kf.transitionMatrix=np.eye(9,dtype=np.float32)
        q=np.zeros((9,9),dtype=np.float32)
        q[0:3,0:3]=np.eye(3)*1e-3; q[3:6,3:6]=np.eye(3)*1e-1; q[6:9,6:9]=np.eye(3)*1e-1
        self.kf.processNoiseCov=q
        self.kf.measurementNoiseCov=np.eye(3,dtype=np.float32)*1e-2
        self.initialized=False; self.prev_time=None; self._reset_cov()

    def _reset_cov(self):
        p=np.eye(9,dtype=np.float32); p[3:6,3:6]*=10.0; p[6:9,6:9]*=100.0
        self.kf.errorCovPost=p

    def _build_F(self, dt):
        F=np.eye(9,dtype=np.float32)
        for i in range(3):
            F[i,i+3]=dt; F[i,i+6]=0.5*dt*dt; F[i+3,i+6]=dt
        return F

    def _init(self, x, y, z, t):
        self._reset_cov()
        self.kf.statePost=np.array([[x],[y],[z],[0],[0],[0],[0],[0],[0]],dtype=np.float32)
        self.initialized=True; self.prev_time=t

    def update(self, x, y, z):
        now=time.time()
        if not self.initialized:
            self._init(x,y,z,now); return x,y,z,0.,0.,0.,0.
        dt=max(now-self.prev_time,1e-3); self.prev_time=now
        self.kf.transitionMatrix=self._build_F(dt)
        pred=self.kf.predict()
        meas=np.array([[x],[y],[z]],dtype=np.float32)
        innov=float(np.linalg.norm(meas.flatten()-pred[:3].flatten()))
        corr=self.kf.correct(meas)
        return float(corr[0]),float(corr[1]),float(corr[2]),\
               float(corr[3]),float(corr[4]),float(corr[5]),innov

    def reset(self):
        self.initialized=False; self.prev_time=None
        self.kf.statePost=np.zeros((9,1),dtype=np.float32); self._reset_cov()

    @property
    def model_info(self): return "CA"


class KFModelCTRV:
    """
    선회율 7D EKF: 상태 [x, y, z, v, yaw, yaw_rate, vz]
    비선형 전이 → Extended Kalman Filter, 직선/선회 자동 분기
    """
    _EPS=1e-4

    def __init__(self):
        self.n=7; self.m=3
        self.Q=np.diag([1e-3,1e-3,1e-3,1e-1,1e-2,1e-2,1e-1]).astype(np.float64)
        self.R=np.eye(3,dtype=np.float64)*1e-2
        self.H=np.zeros((3,7),dtype=np.float64)
        self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
        self.x=np.zeros((7,1),dtype=np.float64)
        self.P=self._init_P()
        self.initialized=False; self.prev_time=None

    def _init_P(self):
        P=np.eye(7,dtype=np.float64)
        P[3,3]=10.0; P[4,4]=np.pi; P[5,5]=1.0; P[6,6]=10.0
        return P

    def _f(self, x, dt):
        px,py,pz,v,yaw,yr,vz=x.flatten()
        if abs(yr)<self._EPS:
            nx=px+v*math.cos(yaw)*dt; ny=py+v*math.sin(yaw)*dt
        else:
            nx=px+(v/yr)*(math.sin(yaw+yr*dt)-math.sin(yaw))
            ny=py+(v/yr)*(-math.cos(yaw+yr*dt)+math.cos(yaw))
        return np.array([[nx],[ny],[pz+vz*dt],[v],[yaw+yr*dt],[yr],[vz]],dtype=np.float64)

    def _jacobian(self, x, dt):
        _,_,_,v,yaw,yr,_=x.flatten()
        F=np.eye(7,dtype=np.float64)
        if abs(yr)<self._EPS:
            F[0,3]=math.cos(yaw)*dt;  F[0,4]=-v*math.sin(yaw)*dt
            F[1,3]=math.sin(yaw)*dt;  F[1,4]=v*math.cos(yaw)*dt
            F[2,6]=dt; F[4,5]=dt
        else:
            sy=math.sin(yaw); cy=math.cos(yaw)
            syt=math.sin(yaw+yr*dt); cyt=math.cos(yaw+yr*dt)
            F[0,3]=(syt-sy)/yr;         F[0,4]=(v/yr)*(cyt-cy)
            F[0,5]=v*(dt*cyt/yr-(syt-sy)/(yr*yr))
            F[1,3]=(-cyt+cy)/yr;        F[1,4]=(v/yr)*(syt-sy)
            F[1,5]=v*(dt*syt/yr+(cyt-cy)/(yr*yr))
            F[2,6]=dt; F[4,5]=dt
        return F

    def _init(self, x, y, z, t):
        self.P=self._init_P()
        self.x=np.array([[x],[y],[z],[0],[0],[0],[0]],dtype=np.float64)
        self.initialized=True; self.prev_time=t

    def update(self, x, y, z):
        now=time.time()
        if not self.initialized:
            self._init(x,y,z,now); return x,y,z,0.,0.,0.,0.
        dt=max(now-self.prev_time,1e-3); self.prev_time=now
        x_pred=self._f(self.x,dt)
        Fj=self._jacobian(self.x,dt)
        P_pred=Fj@self.P@Fj.T+self.Q
        meas=np.array([[x],[y],[z]],dtype=np.float64)
        innov_vec=meas-self.H@x_pred
        innov=float(np.linalg.norm(innov_vec))
        S=self.H@P_pred@self.H.T+self.R
        try:
            # [C3 수정] S 특이행렬 보호 — inv 대신 solve 사용 (수치 안정성↑)
            K=P_pred@self.H.T@np.linalg.solve(S.T,np.eye(self.m)).T
        except np.linalg.LinAlgError:
            # S가 특이행렬이면 예측값만 유지하고 보정 건너뜀
            self.x=x_pred; self.P=P_pred
            fx,fy,fz=float(x_pred[0]),float(x_pred[1]),float(x_pred[2])
            v=float(x_pred[3]); yaw=float(x_pred[4]); vz=float(x_pred[6])
            return fx,fy,fz,v*math.cos(yaw),v*math.sin(yaw),vz,innov
        self.x=x_pred+K@innov_vec
        # [C3 추가] Joseph 형식 — 수치 오차로 P가 음정치(negative definite) 되는 것 방지
        IKH=np.eye(7)-K@self.H
        self.P=IKH@P_pred@IKH.T+K@self.R@K.T
        self.x[4,0]=math.atan2(math.sin(self.x[4,0]),math.cos(self.x[4,0]))
        fx=float(self.x[0]); fy=float(self.x[1]); fz=float(self.x[2])
        v=float(self.x[3]); yaw=float(self.x[4]); vz=float(self.x[6])
        return fx,fy,fz,v*math.cos(yaw),v*math.sin(yaw),vz,innov

    def reset(self):
        self.initialized=False; self.prev_time=None
        self.x=np.zeros((7,1),dtype=np.float64); self.P=self._init_P()

    @property
    def model_info(self):
        if not self.initialized: return "CTRV"
        yr=float(self.x[5])
        m="직선" if abs(yr)<self._EPS else f"선회({math.degrees(yr):.1f}°/s)"
        return f"CTRV[{m}]"


class KFModelIMM:
    """IMM: CV + CTRV 확률 가중 혼합, 직선↔선회 자동 전환"""
    def __init__(self):
        self.filters=[KFModelCV(),KFModelCTRV()]
        self.mu=np.array([0.5,0.5])
        self.PI=np.array([[0.95,0.05],[0.05,0.95]])
        self.initialized=False

    def _like(self, innov, s=0.05):
        return math.exp(-0.5*(innov/s)**2)+1e-300

    def update(self, x, y, z):
        if not self.initialized:
            for f in self.filters: f.update(x,y,z)
            self.initialized=True; return x,y,z,0.,0.,0.,0.
        results=[f.update(x,y,z) for f in self.filters]
        innovs=[r[6] for r in results]
        c=self.PI.T@self.mu
        L=np.array([self._like(i) for i in innovs])
        self.mu=L*c; self.mu/=(self.mu.sum()+1e-300)
        fx=sum(self.mu[i]*results[i][0] for i in range(2))
        fy=sum(self.mu[i]*results[i][1] for i in range(2))
        fz=sum(self.mu[i]*results[i][2] for i in range(2))
        vx=sum(self.mu[i]*results[i][3] for i in range(2))
        vy=sum(self.mu[i]*results[i][4] for i in range(2))
        vz=sum(self.mu[i]*results[i][5] for i in range(2))
        return fx,fy,fz,vx,vy,vz,float(np.dot(self.mu,innovs))

    def reset(self):
        for f in self.filters: f.reset()
        self.mu=np.array([0.5,0.5]); self.initialized=False

    @property
    def model_info(self):
        return f"IMM CV:{self.mu[0]:.2f} CTRV:{self.mu[1]:.2f}"


def create_tracker(name):
    return {'cv':KFModelCV,'ca':KFModelCA,'ctrv':KFModelCTRV,'imm':KFModelIMM}[name]()


# ─────────────────────────────────────────────────────────────────────
# [3] ByteTrack — 다중 객체 추적
# ─────────────────────────────────────────────────────────────────────
class KalmanBoxFilter:
    """ByteTrack용 2D 바운딩박스 칼만 필터 (numpy 구현)
    상태: [cx, cy, a, h, vcx, vcy, va, vh]  a=w/h 종횡비"""
    _W_POS=1/20; _W_VEL=1/160

    def initiate(self, meas):
        h=meas[3]
        std=[2*self._W_POS*h,2*self._W_POS*h,1e-2,2*self._W_POS*h,
             10*self._W_VEL*h,10*self._W_VEL*h,1e-5,10*self._W_VEL*h]
        return np.concatenate([meas,np.zeros(4)]), np.diag(np.square(std))

    def predict(self, mean, cov):
        h=mean[3]
        std=[self._W_POS*h,self._W_POS*h,1e-2,self._W_POS*h,
             self._W_VEL*h,self._W_VEL*h,1e-5,self._W_VEL*h]
        F=np.eye(8); F[0,4]=F[1,5]=F[2,6]=F[3,7]=1.0
        return F@mean, F@cov@F.T+np.diag(np.square(std))

    def update(self, mean, cov, meas):
        h=max(mean[3], 1.0)  # [C4 수정] h≈0 방지 → R/S 특이행렬 예방
        R=np.diag(np.square([self._W_POS*h,self._W_POS*h,1e-1,self._W_POS*h]))
        H=np.eye(4,8); S=H@cov@H.T+R
        try:
            K=cov@H.T@np.linalg.solve(S.T,np.eye(4)).T
        except np.linalg.LinAlgError:
            return mean, cov  # S 특이행렬이면 업데이트 건너뜀
        new_mean=mean+K@(meas-H@mean)
        IKH=np.eye(8)-K@H
        new_cov=IKH@cov@IKH.T+K@R@K.T  # Joseph 형식
        return new_mean, new_cov


class STrack:
    """ByteTrack 단일 트랙: new → tracked(2f확인) → lost → 제거"""
    _kbf=KalmanBoxFilter()

    def __init__(self, det, track_id, frame_id):
        self.track_id=track_id; self.score=float(det[4])
        self.state='new'; self.last_frame=frame_id; self.tracklet_len=0
        x1,y1,x2,y2=det[:4]
        self._tlwh=np.array([x1,y1,x2-x1,y2-y1],dtype=np.float64)
        self._mean,self._cov=self._kbf.initiate(self._to_xyah())

    def _to_xyah(self):
        w,h=self._tlwh[2],max(self._tlwh[3],1)
        return np.array([self._tlwh[0]+w/2,self._tlwh[1]+h/2,w/h,h])

    def predict(self):
        self._mean,self._cov=self._kbf.predict(self._mean,self._cov)

    def update(self, det, frame_id):
        x1,y1,x2,y2=det[:4]
        self._tlwh=np.array([x1,y1,x2-x1,y2-y1],dtype=np.float64)
        self.score=float(det[4])
        self._mean,self._cov=self._kbf.update(self._mean,self._cov,self._to_xyah())
        self.last_frame=frame_id; self.tracklet_len+=1; self.state='tracked'

    @property
    def is_confirmed(self): return self.tracklet_len>=2

    @property
    def tlbr(self):
        cx,cy,a,h=self._mean[:4]; w=a*h
        return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])

    @property
    def center(self):
        return int(self._mean[0]),int(self._mean[1])


class ByteTracker:
    """
    ByteTrack 다중 객체 추적기 — 3단계 IoU 매칭
      1차: 활성 트랙  ↔ 고신뢰도 검출 (score ≥ track_thresh)
      2차: 미매칭 트랙 ↔ 저신뢰도 검출 (second_thresh ≤ score < track_thresh)
      3차: 소실 트랙  ↔ 남은 고신뢰도 검출 (재진입 처리)
    """
    def __init__(self, track_thresh=0.6, second_thresh=0.3, match_thresh=0.8, max_lost=30):
        self.track_thresh=track_thresh; self.second_thresh=second_thresh
        self.match_thresh=match_thresh; self.max_lost=max_lost
        self.tracked=[]; self.lost=[]; self.frame_id=0; self._next_id=1

    @staticmethod
    def _iou(b1, b2):
        x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
        x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)

    def _associate(self, tracks, dets, iou_thresh):
        if not tracks or not dets:
            # 어느 쪽이 비어있어도: 매칭=없음, 미매칭트랙=전부, 미매칭검출=전부
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost=np.array([[1.0-self._iou(t.tlbr,d[:4]) for d in dets] for t in tracks])
        if SCIPY_AVAILABLE:
            ri,ci=linear_sum_assignment(cost)
            pairs=[(r,c) for r,c in zip(ri,ci) if cost[r,c]<=1-iou_thresh]
        else:
            pairs,used_r,used_c=[],set(),set()
            for cv,r,c in sorted([(cost[r,c],r,c) for r in range(len(tracks)) for c in range(len(dets))]):
                if cv>1-iou_thresh: break
                if r not in used_r and c not in used_c:
                    pairs.append((r,c)); used_r.add(r); used_c.add(c)
        mr={r for r,c in pairs}; mc={c for r,c in pairs}
        return pairs,[i for i in range(len(tracks)) if i not in mr],[j for j in range(len(dets)) if j not in mc]

    def update(self, detections):
        self.frame_id+=1
        high=[d for d in detections if d[4]>=self.track_thresh]
        low =[d for d in detections if self.second_thresh<=d[4]<self.track_thresh]
        for t in self.tracked+self.lost: t.predict()

        m1,ut1,ud1=self._associate(self.tracked,high,self.match_thresh)
        for ti,di in m1: self.tracked[ti].update(high[di],self.frame_id)

        rem=[ self.tracked[i] for i in ut1]
        m2,ut2,_=self._associate(rem,low,0.5)
        for ti,di in m2: rem[ti].update(low[di],self.frame_id)
        for i in ut2: rem[i].state='lost'
        newly_lost=[rem[i] for i in ut2]

        rem_high=[high[i] for i in ud1]
        m3,_,ud3=self._associate(self.lost,rem_high,0.5)
        # 소실 트랙 중 재매칭된 것은 별도 수집 후 state='tracked' 복귀
        reactivated=[]
        for ti,di in m3:
            self.lost[ti].update(rem_high[di],self.frame_id)
            reactivated.append(self.lost[ti])   # state가 'tracked'로 바뀜

        new_tracks=[STrack(rem_high[i],self._next_id+k,self.frame_id) for k,i in enumerate(ud3)]
        self._next_id+=len(new_tracks)

        # self.lost 갱신: 재매칭(state='tracked')된 것은 이미 reactivated로 분리됨
        self.lost=[t for t in self.lost+newly_lost
                   if self.frame_id-t.last_frame<=self.max_lost and t.state=='lost']
        # self.tracked 갱신: 기존 tracked + 재활성 + 신규
        self.tracked=([t for t in self.tracked if t.state=='tracked']
                      +reactivated+new_tracks)
        return [t for t in self.tracked if t.is_confirmed]


# ─────────────────────────────────────────────────────────────────────
# [4] LSTM 기반 미래 위치 예측기
# ─────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, out_steps):
            super().__init__()
            self.lstm=nn.LSTM(3,64,num_layers=2,batch_first=True,dropout=0.1)
            self.fc=nn.Linear(64,out_steps*3); self.out_steps=out_steps
        def forward(self, x):
            h,_=self.lstm(x)
            return self.fc(h[:,-1,:]).view(-1,self.out_steps,3)

class LSTMPredictor:
    """온라인 학습 LSTM 궤적 예측기. PyTorch 없으면 선형 외삽으로 대체."""
    SEQ_LEN=30; TRAIN_EVERY=15; EPOCHS=5

    def __init__(self, steps):
        self.steps=steps; self.buf=deque(maxlen=120)
        self.ref=None; self.cnt=0; self.ready=False
        if TORCH_AVAILABLE:
            self.net=_LSTMNet(steps)
            self.opt=optim.Adam(self.net.parameters(),lr=1e-3)
            self.crit=nn.MSELoss()
        else:
            self.net=None; print("ℹ️ LSTM → 선형 외삽 대체")

    def add(self, x, y, z):
        if self.ref is None: self.ref=np.array([x,y,z])
        self.buf.append(np.array([x,y,z])-self.ref); self.cnt+=1
        if len(self.buf)>=self.SEQ_LEN+self.steps and self.cnt%self.TRAIN_EVERY==0:
            self._train(); self.ready=True

    def _train(self):
        if not TORCH_AVAILABLE: return
        b=np.array(list(self.buf),dtype=np.float32)
        xs,ys=[],[]
        for i in range(len(b)-self.SEQ_LEN-self.steps+1):
            xs.append(b[i:i+self.SEQ_LEN]); ys.append(b[i+self.SEQ_LEN:i+self.SEQ_LEN+self.steps])
        if not xs: return
        X,Y=torch.tensor(np.stack(xs)),torch.tensor(np.stack(ys))
        self.net.train()
        # [C6] 메인루프 블로킹 학습 — 소요시간 측정 후 경고
        t0=time.time()
        for _ in range(self.EPOCHS):
            self.opt.zero_grad(); loss=self.crit(self.net(X),Y); loss.backward(); self.opt.step()
        elapsed=(time.time()-t0)*1000
        if elapsed>50:
            print(f"⚠️ [LSTM] 학습 {elapsed:.0f}ms — 프레임 드롭 주의 (EPOCHS 줄이기 권장)")

    def predict(self):
        if len(self.buf)<self.SEQ_LEN or self.ref is None: return None
        if not self.ready or not TORCH_AVAILABLE:
            tail=np.array(list(self.buf)[-5:])
            vel=(tail[-1]-tail[0])/max(len(tail)-1,1)
            return [tuple((tail[-1]+vel*k+self.ref).tolist()) for k in range(1,self.steps+1)]
        seq=np.array(list(self.buf)[-self.SEQ_LEN:],dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            out=self.net(torch.tensor(seq).unsqueeze(0)).squeeze(0).numpy()
        return [(float(p[0]+self.ref[0]),float(p[1]+self.ref[1]),float(p[2]+self.ref[2])) for p in out]

    def reset(self):
        self.buf.clear(); self.ref=None; self.cnt=0; self.ready=False


# ─────────────────────────────────────────────────────────────────────
# [5] MOTP 평가기
# ─────────────────────────────────────────────────────────────────────
class MOTPEvaluator:
    """MOTP ≈ Σ(innovation_norm) / Σ(count). 값이 작을수록 추적 정밀도 높음."""
    def __init__(self):
        self.total_d=0.0; self.total_c=0; self.frame=0; self._log=[]

    _LOG_MAX=18000  # 최대 보관 항목 수 (30fps×10분=18000)

    def update(self, innov):
        self.total_d+=innov; self.total_c+=1; self.frame+=1
        self._log.append((self.frame,round(innov*100,3)))
        # [C5 수정] 무제한 메모리 성장 방지 — 초과 시 앞부분 절반 삭제
        if len(self._log)>self._LOG_MAX:
            self._log=self._log[self._LOG_MAX//2:]

    @property
    def motp_m(self): return self.total_d/self.total_c if self.total_c else 0.0

    def text(self): return f"MOTP:{self.motp_m*100:.2f}cm (N={self.total_c})"

    def save(self, path='motp_log.csv'):
        with open(path,'w',newline='') as f:
            csv.writer(f).writerows([['frame','innovation_cm']]+self._log)
        print(f"📊 MOTP 로그 → {path}")


# ─────────────────────────────────────────────────────────────────────
# [6] 하드웨어 초기화 (전체 시작 시간 측정)
# ─────────────────────────────────────────────────────────────────────
_startup_begin = time.time()

# TensorRT 엔진 로드 (내부에서 진행 메시지 출력)
trt_brain = JetsonTRTEngine('best.engine')

# GPU 워밍업 — 첫 번째 실제 추론의 JIT 컴파일 지연 제거
# 빈 더미 이미지로 1회 추론 → GPU 커널 사전 컴파일
print("⏳ GPU 워밍업 중 (첫 프레임 지연 제거)...")
_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
trt_brain.infer(_dummy)
print("✅ GPU 워밍업 완료")

# RealSense 파이프라인 시작 (USB 장치 열거 + 센서 초기화, 1~3초 소요)
print("⏳ RealSense D435i 초기화 중...")
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile  = pipeline.start(config)
align    = rs.align(rs.stream.color)
print("✅ RealSense 초기화 완료")

# 깊이 후처리 필터 체인
spatial_filter  = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_fill       = rs.hole_filling_filter()

# intrinsics: 세션 중 불변 → 루프 밖에서 1회만 취득
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

DEPTH_MIN_M = 0.15   # RealSense D435i 최소 신뢰 거리
DEPTH_MAX_M = 8.0    # 정밀 착륙 유효 고도 상한

_startup_sec = time.time() - _startup_begin
print(f"🚀 [성공] 전체 초기화 완료 — 총 소요시간: {_startup_sec:.1f}초")

# ─────────────────────────────────────────────────────────────────────
# [7] 인스턴스 생성
# ─────────────────────────────────────────────────────────────────────
single_kf    = create_tracker(args.model)
byte_tracker = ByteTracker() if args.tracker=='bytetrack' else None
track_3d_kfs = {}   # {track_id: KFModel3D}

lstm_pred = LSTMPredictor(args.predict) if args.predict>0 else None
motp_eval = MOTPEvaluator()             if args.motp       else None


# ─────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────
def get_median_depth(depth_frame, cx, cy, half=2):
    """5×5 영역 유효 깊이 중앙값 (홀 픽셀 제외)"""
    # get_distance()는 int 인수만 허용 — float 전달 시 TypeError 방지
    cx, cy = int(cx), int(cy)
    vals=[depth_frame.get_distance(cx+dx, cy+dy)
          for dy in range(-half, half+1) for dx in range(-half, half+1)
          if 0<=cx+dx<640 and 0<=cy+dy<480]
    vals=[v for v in vals if v>0]
    return float(np.median(vals)) if vals else 0.0


def fov_angles(cx, cy):
    """깊이 없을 때 FOV 기반 2D 픽셀 각도 계산 (원본 로직 보존)"""
    dx=cx-intrinsics.ppx; dy=cy-intrinsics.ppy
    return math.atan2(dx,intrinsics.fx), math.atan2(dy,intrinsics.fy)


def project_to_pixel(fx, fy, fz):
    """3D 미터 → 2D 픽셀"""
    if fz<=0: return None
    px=int(fx*intrinsics.fx/fz+intrinsics.ppx)
    py=int(fy*intrinsics.fy/fz+intrinsics.ppy)
    return (px,py) if 0<=px<640 and 0<=py<480 else None


def track_color(tid):
    """트랙 ID → 고유 BGR 색상 (독립 RNG로 전역 난수 상태 오염 방지)"""
    rng=np.random.RandomState(tid*17%256)
    return tuple(int(c) for c in rng.randint(80,255,3))


def draw_info(img, x1, y_top, fx, fy, fz, vx, vy, vz, speed, color=(255,255,0)):
    """위치·속도 텍스트 4줄 오버레이"""
    y=max(y_top,70)
    cv2.putText(img,f"Offset X:{fx:.2f}m  Y:{fy:.2f}m",
                (x1,y-55),cv2.FONT_HERSHEY_SIMPLEX,0.42,color,1)
    cv2.putText(img,f"Alt(Z): {fz:.2f}m",
                (x1,y-40),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,0),1)
    cv2.putText(img,f"Vel Vx:{vx:.2f} Vy:{vy:.2f} Vz:{vz:.2f} m/s",
                (x1,y-25),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,200,255),1)
    cv2.putText(img,f"Speed:{speed:.2f}m/s [{speed*100:.0f}cm/s]",
                (x1,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,140,255),1)


def send_mavlink(angle_x, angle_y, distance):
    """MAVLink LANDING_TARGET 10Hz 송신 (MAVLink 1/2 호환)"""
    global _last_mav_send
    now=time.time()
    if master is None or now-_last_mav_send<1.0/MAVLINK_SEND_HZ: return
    _last_mav_send=now
    frame=getattr(mavutil.mavlink,'MAV_FRAME_BODY_NED',8)
    try:
        # MAVLink 2 (최신 pymavlink)
        master.mav.landing_target_send(
            time_usec=int(now*1e6), target_num=0, frame=frame,
            angle_x=angle_x, angle_y=angle_y, distance=distance,
            size_x=0.0, size_y=0.0,
            type=2, position_valid=0)
    except TypeError:
        try:
            # MAVLink 1 (구형 pymavlink 8인자)
            master.mav.landing_target_send(
                int(now*1e6),0,frame,angle_x,angle_y,distance,0.0,0.0)
        except Exception as mav_err:
            # [C2 수정] 네트워크 오류(OSError, 소켓 끊김 등) 메인루프 크래시 방지
            print(f"⚠️ MAVLink 송신 실패 (MAVLink1): {mav_err}")
            return
    except Exception as mav_err:
        # [C2 수정] MAVLink2 전송 중 네트워크/소켓 예외 처리
        print(f"⚠️ MAVLink 송신 실패 (MAVLink2): {mav_err}")
        return
    # 송신 성공 시 각도/거리 출력
    print(f"📡 [MAVLink] LANDING_TARGET 송출 -> "
          f"angle_x:{math.degrees(angle_x):.2f}°, "
          f"angle_y:{math.degrees(angle_y):.2f}°, "
          f"Dist:{distance:.2f}m")


# ─────────────────────────────────────────────────────────────────────
# [8] 메인 루프
# ─────────────────────────────────────────────────────────────────────
try:
    while True:
        frames=pipeline.wait_for_frames()
        aligned_frames=align.process(frames)
        depth_frame=aligned_frames.get_depth_frame()
        color_frame=aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 깊이 후처리 필터 체인
        # 각 필터 출력은 rs.frame 타입 → 매 단계 as_depth_frame() 명시 변환
        # (마지막 단계에만 적용하면 intermediate frame이 depth_frame 타입을 잃음)
        depth_frame=spatial_filter.process(depth_frame).as_depth_frame()
        depth_frame=temporal_filter.process(depth_frame).as_depth_frame()
        depth_frame=hole_fill.process(depth_frame).as_depth_frame()

        color_image=np.asanyarray(color_frame.get_data())

        # [M1] TensorRT 추론 → numpy 벡터화로 8400개 필터링
        # 원본: Python for 루프 8400회 → ~5~15ms 지연
        # 수정: numpy 마스킹+슬라이싱 → ~0.1ms
        raw_out = trt_brain.infer(color_image).reshape(5, 8400).T  # (8400, 5)
        scores  = raw_out[:, 4]
        vmask   = scores >= 0.3          # ByteTrack 저신뢰도 포함 최소 임계값
        vpreds  = raw_out[vmask]         # (N, 5) — score 0.3 이상 박스만

        if len(vpreds):
            xc  = vpreds[:, 0];  yc = vpreds[:, 1]
            bw  = vpreds[:, 2];  bh = vpreds[:, 3]
            sc  = vpreds[:, 4]
            x1s = (xc - bw / 2).astype(int)
            y1s = ((yc - bh / 2) * (480 / 640)).astype(int)
            x2s = (xc + bw / 2).astype(int)
            y2s = ((yc + bh / 2) * (480 / 640)).astype(int)
            # column_stack 시 sc(float)로 좌표가 float 업캐스팅되는 문제 방지
            # → 좌표는 int, score는 float로 명시 분리
            all_dets = [
                [int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i]), float(sc[i])]
                for i in range(len(vpreds))
            ]
        else:
            all_dets = []

        # ══════════════════════════════════════════════════════════════
        # 모드 A: single — 최고 신뢰도 단일 객체
        # ══════════════════════════════════════════════════════════════
        if args.tracker=='single':
            best=max(all_dets,key=lambda d:d[4]) if all_dets else None

            if best and best[4]>=0.6:
                x1,y1,x2,y2,score=best
                cx,cy=(x1+x2)//2,(y1+y2)//2

                # [C7 수정] cx,cy 클램프 → 경계 밖 검출도 FOV 각도 계산 및 MAVLink 송신
                cx_c=max(0,min(639,cx)); cy_c=max(0,min(479,cy))
                angle_x,angle_y=fov_angles(cx_c,cy_c)
                dv=get_median_depth(depth_frame,cx_c,cy_c)

                cv2.rectangle(color_image,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(color_image,(cx_c,cy_c),5,(0,0,255),-1)

                if DEPTH_MIN_M<dv<DEPTH_MAX_M:
                    # 3D 역투영 + 칼만 필터
                    rx,ry,rz=rs.rs2_deproject_pixel_to_point(intrinsics,[cx_c,cy_c],dv)
                    fx,fy,fz,vx,vy,vz,innov=single_kf.update(rx,ry,rz)
                    speed=math.sqrt(vx**2+vy**2+vz**2)

                    # 3D 기반 각도로 덮어쓰기 (더 정확)
                    angle_x=math.atan2(fx,fz); angle_y=math.atan2(fy,fz)

                    if motp_eval: motp_eval.update(innov)

                    # LSTM
                    futures=None
                    if lstm_pred:
                        lstm_pred.add(fx,fy,fz)
                        futures=lstm_pred.predict()

                    draw_info(color_image,x1,y1,fx,fy,fz,vx,vy,vz,speed)
                    cv2.putText(color_image,f"Model:{single_kf.model_info}",
                                (5,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)

                    if futures:
                        for k,(px_f,py_f,pz_f) in enumerate(futures):
                            pix=project_to_pixel(px_f,py_f,pz_f)
                            if pix:
                                a=1.0-k/len(futures)
                                cv2.circle(color_image,pix,max(2,int(5*a)),
                                           (int(255*a),int(100*a),255),-1)

                    print(f"🎯 [{single_kf.model_info}] "
                          f"X:{fx*100:.1f} Y:{fy*100:.1f} Z:{fz*100:.1f}cm | "
                          f"V:{speed*100:.1f}cm/s | innov:{innov*100:.2f}cm"
                          +(f" | {motp_eval.text()}" if motp_eval else ""))
                else:
                    # 깊이 미확정 → FOV 각도만 표시 (원본 동작 보존)
                    cv2.putText(color_image,"Depth: N/A (Too Close/Far)",
                                (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                    print(f"🎯 패드 포착(깊이 미확정) -> "
                          f"angle_x:{math.degrees(angle_x):.1f}° "
                          f"angle_y:{math.degrees(angle_y):.1f}°")
                    single_kf.reset()
                    if lstm_pred: lstm_pred.reset()

                # MAVLink: 깊이 있으면 3D 각도, 없으면 FOV 각도로 항상 송신
                send_mavlink(angle_x,angle_y,dv)

                if motp_eval:
                    cv2.putText(color_image,motp_eval.text(),
                                (5,470),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,255,180),1)
            else:
                single_kf.reset()
                if lstm_pred: lstm_pred.reset()

        # ══════════════════════════════════════════════════════════════
        # 모드 B: bytetrack — 다중 객체 ByteTrack + 트랙별 3D KF
        # ══════════════════════════════════════════════════════════════
        else:
            active=byte_tracker.update(all_dets)
            primary=None   # (score, tid, fx,fy,fz, vx,vy,vz, innov, angle_x, angle_y, dv)

            for track in active:
                tid=track.track_id
                cx,cy=track.center
                if not (0<=cx<640 and 0<=cy<480): continue

                # FOV 각도 항상 계산
                angle_x,angle_y=fov_angles(cx,cy)
                dv=get_median_depth(depth_frame,cx,cy)

                col=track_color(tid)
                tlbr=track.tlbr.astype(int)
                cv2.rectangle(color_image,(tlbr[0],tlbr[1]),(tlbr[2],tlbr[3]),col,2)
                cv2.putText(color_image,f"ID{tid} {track.score:.2f}",
                            (tlbr[0],tlbr[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)

                if DEPTH_MIN_M<dv<DEPTH_MAX_M:
                    rx,ry,rz=rs.rs2_deproject_pixel_to_point(intrinsics,[cx,cy],dv)
                    if tid not in track_3d_kfs:
                        track_3d_kfs[tid]=create_tracker(args.model)
                    fx,fy,fz,vx,vy,vz,innov=track_3d_kfs[tid].update(rx,ry,rz)
                    speed=math.sqrt(vx**2+vy**2+vz**2)
                    angle_x=math.atan2(fx,fz); angle_y=math.atan2(fy,fz)

                    if motp_eval: motp_eval.update(innov)
                    draw_info(color_image,tlbr[0],tlbr[1],fx,fy,fz,vx,vy,vz,speed,col)

                    if primary is None or track.score>primary[0]:
                        primary=(track.score,tid,fx,fy,fz,vx,vy,vz,innov,angle_x,angle_y,dv)
                else:
                    cv2.putText(color_image,"Depth:N/A",
                                (tlbr[0],tlbr[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
                    # 깊이 없어도 FOV 각도로 주 트랙 후보
                    if primary is None or track.score>primary[0]:
                        primary=(track.score,tid,0,0,0,0,0,0,0,angle_x,angle_y,0.0)

            if primary:
                sc,ptid,fx,fy,fz,vx,vy,vz,innov,angle_x,angle_y,dv=primary

                # LSTM — 주 트랙만 적용
                if lstm_pred and dv>0:
                    lstm_pred.add(fx,fy,fz)
                    futures=lstm_pred.predict()
                    if futures:
                        for k,(px_f,py_f,pz_f) in enumerate(futures):
                            pix=project_to_pixel(px_f,py_f,pz_f)
                            if pix:
                                a=1.0-k/len(futures)
                                cv2.circle(color_image,pix,max(2,int(5*a)),
                                           (int(255*a),int(100*a),255),-1)

                send_mavlink(angle_x,angle_y,dv)
                pos_str=(f"X:{fx*100:.1f} Y:{fy*100:.1f} Z:{fz*100:.1f}cm"
                         if dv>0 else "깊이미확정(FOV각도로송신)")
                print(f"[ByteTrack|{args.model.upper()}] 활성:{len(active)}개 | "
                      f"주트랙ID:{ptid} score:{sc:.2f} | {pos_str}"
                      +(f" | {motp_eval.text()}" if motp_eval else ""))
            else:
                if lstm_pred: lstm_pred.reset()

            # 소멸 트랙 3D KF 정리
            active_ids={t.track_id for t in active}
            for tid in list(track_3d_kfs):
                if tid not in active_ids:
                    track_3d_kfs[tid].reset(); del track_3d_kfs[tid]

            cv2.putText(color_image,
                        f"ByteTrack | {args.model.upper()} | 활성:{len(active)}",
                        (5,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
            if motp_eval:
                cv2.putText(color_image,motp_eval.text(),
                            (5,470),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,255,180),1)

        # ── 공통 출력 ─────────────────────────────────────────────────
        if ENABLE_STREAMING and out is not None and out.isOpened():
            out.write(color_image)
        cv2.imshow("Jetson Local View",color_image)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

# ─────────────────────────────────────────────────────────────────────
# 안전 종료 및 자원 해제
# ─────────────────────────────────────────────────────────────────────
except KeyboardInterrupt:
    print("\n사용자에 의해 프로그램을 종료합니다.")
except Exception as e:
    print(f"\n메인 루프 실행 중 오류 발생: {e}")
finally:
    print("시스템 자원을 해제합니다...")
    if motp_eval and motp_eval.total_c>0:
        motp_eval.save()
        print(f"📊 최종 {motp_eval.text()}")
    pipeline.stop()
    cv2.destroyAllWindows()
    if ENABLE_STREAMING and out is not None and out.isOpened():
        out.release()
