#!/usr/bin/env python3
"""
jetson_inference_advanced.py — Jetson 정밀 착륙 비전 시스템 고도화

실행 예:
  python3 jetson_inference_advanced.py --model ctrv --tracker bytetrack --predict 15 --motp
  python3 jetson_inference_advanced.py --model imm  --tracker bytetrack
  python3 jetson_inference_advanced.py --model ca   --tracker single --motp

인수:
  --model   {cv,ca,ctrv,imm}      칼만 필터 모델 (기본: cv)
  --tracker {single,bytetrack}    추적 방식 (기본: single)
  --predict N                     LSTM 미래 N 스텝 예측 (0=비활성)
  --motp                          MOTP 추적 정밀도 평가 출력
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
parser = argparse.ArgumentParser(description='Jetson 정밀 착륙 비전 시스템 고도화')
parser.add_argument('--model', choices=['cv', 'ca', 'ctrv', 'imm'], default='cv',
                    help='칼만 필터 모델 (기본: cv)\n'
                         '  cv   : 등속도 6D 선형\n'
                         '  ca   : 등가속도 9D 선형\n'
                         '  ctrv : 선회율 7D EKF (비선형)\n'
                         '  imm  : CV+CTRV 확률 혼합')
parser.add_argument('--tracker', choices=['single', 'bytetrack'], default='single',
                    help='추적 방식 (기본: single)\n'
                         '  single    : 최고 신뢰도 단일 객체 추적\n'
                         '  bytetrack : 다중 객체 ByteTrack (ID 유지)')
parser.add_argument('--predict', type=int, default=0, metavar='N',
                    help='LSTM 미래 위치 예측 스텝 수 (0=비활성)')
parser.add_argument('--motp', action='store_true',
                    help='MOTP 추적 정밀도 평가 활성화')
args = parser.parse_args()

print(f"[설정] 모델={args.model.upper()} | 추적={args.tracker} "
      f"| LSTM={args.predict}스텝 | MOTP={args.motp} | PyTorch={TORCH_AVAILABLE}")

# ─────────────────────────────────────────────────────────────────────
# GStreamer 스트리밍
# ─────────────────────────────────────────────────────────────────────
gst_pipeline = (
    "appsrc ! videoconvert ! "
    "video/x-raw,format=I420 ! "
    "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
    "rtph264pay ! "
    "udpsink host=192.168.1.30 port=5600"
)
out = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))
if not out.isOpened():
    print("⚠️ GStreamer 초기화 실패 — 스트리밍 없이 계속합니다.")

# ─────────────────────────────────────────────────────────────────────
# MAVLink 비행 제어기 연결
# ─────────────────────────────────────────────────────────────────────
try:
    master = mavutil.mavlink_connection('udp:127.0.0.1:14551')
    master.wait_heartbeat()
    print("🛸 ArduPilot FC MAVLink 연결 완료!")
except Exception as e:
    print(f"⚠️ FC 연결 실패: {e}")
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
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context  = self.engine.create_execution_context()
        self.cuda_lib = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        self.inputs, self.outputs, self.bindings, self._ptrs = [], [], [], []
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            if shape[0] in (-1, 0):
                shape = (1,3,640,640) if self.engine.binding_is_input(binding) else (1,5,8400)
            dtype    = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = np.empty(shape, dtype=dtype)
            cuda_ptr = ctypes.c_void_p()
            self.cuda_lib.cudaMalloc(ctypes.byref(cuda_ptr), host_mem.nbytes)
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
        q[0:3,0:3] = np.eye(3)*1e-3;  q[3:6,3:6] = np.eye(3)*1e-1
        self.kf.processNoiseCov     = q
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32)*1e-2
        self.initialized = False;  self.prev_time = None;  self._reset_cov()

    def _reset_cov(self):
        p = np.eye(6, dtype=np.float32);  p[3:6,3:6] *= 10.0
        self.kf.errorCovPost = p

    def _init(self, x, y, z, t):
        self._reset_cov()
        self.kf.statePost = np.array([[x],[y],[z],[0],[0],[0]], dtype=np.float32)
        self.initialized = True;  self.prev_time = t

    def update(self, x, y, z):
        now = time.time()
        if not self.initialized:
            self._init(x, y, z, now);  return x, y, z, 0., 0., 0., 0.
        dt = max(now - self.prev_time, 1e-3);  self.prev_time = now
        self.kf.transitionMatrix[0,3] = self.kf.transitionMatrix[1,4] = \
            self.kf.transitionMatrix[2,5] = dt
        pred  = self.kf.predict()
        meas  = np.array([[x],[y],[z]], dtype=np.float32)
        innov = float(np.linalg.norm(meas.flatten() - pred[:3].flatten()))
        corr  = self.kf.correct(meas)
        return float(corr[0]),float(corr[1]),float(corr[2]),\
               float(corr[3]),float(corr[4]),float(corr[5]), innov

    def reset(self):
        self.initialized = False;  self.prev_time = None
        self.kf.statePost = np.zeros((6,1), dtype=np.float32);  self._reset_cov()

    @property
    def model_info(self): return "CV"


class KFModelCA:
    """등가속도 9D 선형 칼만: 상태 [x,y,z, vx,vy,vz, ax,ay,az]"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(9, 3)
        H = np.zeros((3,9), dtype=np.float32)
        H[0,0]=H[1,1]=H[2,2]=1.0
        self.kf.measurementMatrix = H
        self.kf.transitionMatrix  = np.eye(9, dtype=np.float32)
        q = np.zeros((9,9), dtype=np.float32)
        q[0:3,0:3]=np.eye(3)*1e-3; q[3:6,3:6]=np.eye(3)*1e-1; q[6:9,6:9]=np.eye(3)*1e-1
        self.kf.processNoiseCov     = q
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32)*1e-2
        self.initialized = False;  self.prev_time = None;  self._reset_cov()

    def _reset_cov(self):
        p = np.eye(9, dtype=np.float32)
        p[3:6,3:6] *= 10.0;  p[6:9,6:9] *= 100.0
        self.kf.errorCovPost = p

    def _build_F(self, dt):
        F = np.eye(9, dtype=np.float32)
        for i in range(3):
            F[i,i+3]=dt; F[i,i+6]=0.5*dt*dt; F[i+3,i+6]=dt
        return F

    def _init(self, x, y, z, t):
        self._reset_cov()
        self.kf.statePost = np.array([[x],[y],[z],[0],[0],[0],[0],[0],[0]], dtype=np.float32)
        self.initialized = True;  self.prev_time = t

    def update(self, x, y, z):
        now = time.time()
        if not self.initialized:
            self._init(x, y, z, now);  return x, y, z, 0., 0., 0., 0.
        dt = max(now - self.prev_time, 1e-3);  self.prev_time = now
        self.kf.transitionMatrix = self._build_F(dt)
        pred  = self.kf.predict()
        meas  = np.array([[x],[y],[z]], dtype=np.float32)
        innov = float(np.linalg.norm(meas.flatten() - pred[:3].flatten()))
        corr  = self.kf.correct(meas)
        return float(corr[0]),float(corr[1]),float(corr[2]),\
               float(corr[3]),float(corr[4]),float(corr[5]), innov

    def reset(self):
        self.initialized = False;  self.prev_time = None
        self.kf.statePost = np.zeros((9,1), dtype=np.float32);  self._reset_cov()

    @property
    def model_info(self): return "CA"


class KFModelCTRV:
    """
    선회율 7D EKF: 상태 [x, y, z, v, yaw, yaw_rate, vz]
    비선형 전이 → Extended Kalman Filter (야코비안 직접 계산)
    직선(|yaw_rate|<ε) / 선회 두 분기 자동 전환
    """
    _EPS = 1e-4

    def __init__(self):
        self.n = 7;  self.m = 3
        self.Q = np.diag([1e-3,1e-3,1e-3,1e-1,1e-2,1e-2,1e-1]).astype(np.float64)
        self.R = np.eye(3, dtype=np.float64)*1e-2
        self.H = np.zeros((3,7), dtype=np.float64)
        self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
        self.x = np.zeros((7,1), dtype=np.float64)
        self.P = self._init_P()
        self.initialized = False;  self.prev_time = None

    def _init_P(self):
        P = np.eye(7, dtype=np.float64)
        P[3,3]=10.0; P[4,4]=np.pi; P[5,5]=1.0; P[6,6]=10.0
        return P

    def _f(self, x, dt):
        px,py,pz,v,yaw,yr,vz = x.flatten()
        if abs(yr) < self._EPS:
            nx = px + v*math.cos(yaw)*dt
            ny = py + v*math.sin(yaw)*dt
        else:
            nx = px + (v/yr)*(math.sin(yaw+yr*dt)-math.sin(yaw))
            ny = py + (v/yr)*(-math.cos(yaw+yr*dt)+math.cos(yaw))
        return np.array([[nx],[ny],[pz+vz*dt],[v],[yaw+yr*dt],[yr],[vz]], dtype=np.float64)

    def _jacobian(self, x, dt):
        _,_,_,v,yaw,yr,_ = x.flatten()
        F = np.eye(7, dtype=np.float64)
        if abs(yr) < self._EPS:
            F[0,3]= math.cos(yaw)*dt;  F[0,4]=-v*math.sin(yaw)*dt
            F[1,3]= math.sin(yaw)*dt;  F[1,4]= v*math.cos(yaw)*dt
            F[2,6]=dt;  F[4,5]=dt
        else:
            sy=math.sin(yaw); cy=math.cos(yaw)
            syt=math.sin(yaw+yr*dt); cyt=math.cos(yaw+yr*dt)
            F[0,3]=(syt-sy)/yr;         F[0,4]=(v/yr)*(cyt-cy)
            F[0,5]=v*(dt*cyt/yr-(syt-sy)/(yr*yr))
            F[1,3]=(-cyt+cy)/yr;        F[1,4]=(v/yr)*(syt-sy)
            F[1,5]=v*(dt*syt/yr+(cyt-cy)/(yr*yr))
            F[2,6]=dt;  F[4,5]=dt
        return F

    def _init(self, x, y, z, t):
        self.P = self._init_P()
        self.x = np.array([[x],[y],[z],[0],[0],[0],[0]], dtype=np.float64)
        self.initialized=True;  self.prev_time=t

    def update(self, x, y, z):
        now = time.time()
        if not self.initialized:
            self._init(x, y, z, now);  return x, y, z, 0., 0., 0., 0.
        dt = max(now-self.prev_time, 1e-3);  self.prev_time=now
        x_pred = self._f(self.x, dt)
        Fj     = self._jacobian(self.x, dt)
        P_pred = Fj @ self.P @ Fj.T + self.Q
        meas   = np.array([[x],[y],[z]], dtype=np.float64)
        innov_vec = meas - self.H @ x_pred
        innov  = float(np.linalg.norm(innov_vec))
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ innov_vec
        self.P = (np.eye(7)-K @ self.H) @ P_pred
        self.x[4,0] = math.atan2(math.sin(self.x[4,0]), math.cos(self.x[4,0]))
        fx=float(self.x[0]); fy=float(self.x[1]); fz=float(self.x[2])
        v=float(self.x[3]); yaw=float(self.x[4]); vz=float(self.x[6])
        return fx,fy,fz, v*math.cos(yaw),v*math.sin(yaw),vz, innov

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
    """
    IMM: CV + CTRV 확률 가중 혼합
    직선 → CV 우세 / 선회 → CTRV 우세, 자동 전환
    """
    def __init__(self):
        self.filters=[KFModelCV(), KFModelCTRV()]
        self.mu=np.array([0.5,0.5])
        self.PI=np.array([[0.95,0.05],[0.05,0.95]])
        self.initialized=False

    def _like(self, innov, s=0.05):
        return math.exp(-0.5*(innov/s)**2)+1e-300

    def update(self, x, y, z):
        if not self.initialized:
            for f in self.filters: f.update(x,y,z)
            self.initialized=True;  return x,y,z,0.,0.,0.,0.
        results=[f.update(x,y,z) for f in self.filters]
        innovs=[r[6] for r in results]
        c=self.PI.T @ self.mu
        L=np.array([self._like(i) for i in innovs])
        self.mu=L*c;  self.mu/=(self.mu.sum()+1e-300)
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
    cls={'cv':KFModelCV,'ca':KFModelCA,'ctrv':KFModelCTRV,'imm':KFModelIMM}[name]
    return cls()


# ─────────────────────────────────────────────────────────────────────
# [3] ByteTrack — 다중 객체 추적
# ─────────────────────────────────────────────────────────────────────
class KalmanBoxFilter:
    """
    ByteTrack용 2D 바운딩박스 칼만 필터 (순수 numpy 구현)
    상태: [cx, cy, a, h, vcx, vcy, va, vh]   a = w/h 종횡비
    측정: [cx, cy, a, h]
    """
    _W_POS = 1/20
    _W_VEL = 1/160

    def initiate(self, meas):
        """meas: [cx, cy, a, h]"""
        mean = np.concatenate([meas, np.zeros(4)])
        h = meas[3]
        std = [2*self._W_POS*h, 2*self._W_POS*h, 1e-2, 2*self._W_POS*h,
               10*self._W_VEL*h, 10*self._W_VEL*h, 1e-5, 10*self._W_VEL*h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._W_POS*h, self._W_POS*h, 1e-2, self._W_POS*h,
               self._W_VEL*h, self._W_VEL*h, 1e-5, self._W_VEL*h]
        Q = np.diag(np.square(std))
        F = np.eye(8);  F[0,4]=F[1,5]=F[2,6]=F[3,7]=1.0
        return F @ mean, F @ cov @ F.T + Q

    def update(self, mean, cov, meas):
        h = mean[3]
        std = [self._W_POS*h, self._W_POS*h, 1e-1, self._W_POS*h]
        R = np.diag(np.square(std))
        H = np.eye(4, 8)
        S = H @ cov @ H.T + R
        K = cov @ H.T @ np.linalg.inv(S)
        mean = mean + K @ (meas - H @ mean)
        cov  = (np.eye(8) - K @ H) @ cov
        return mean, cov


class STrack:
    """
    ByteTrack 단일 트랙.
    상태: 'new' → 'tracked' (2프레임 확인) → 'lost' → 'removed'
    """
    _kbf = KalmanBoxFilter()

    def __init__(self, det, track_id, frame_id):
        """det: [x1, y1, x2, y2, score]"""
        self.track_id    = track_id
        self.score       = float(det[4])
        self.state       = 'new'
        self.last_frame  = frame_id
        self.tracklet_len = 0
        x1,y1,x2,y2 = det[:4]
        self._tlwh = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float64)
        self._mean, self._cov = self._kbf.initiate(self._to_xyah())

    def _to_xyah(self):
        w, h = self._tlwh[2], max(self._tlwh[3], 1)
        return np.array([self._tlwh[0]+w/2, self._tlwh[1]+h/2, w/h, h])

    def predict(self):
        self._mean, self._cov = self._kbf.predict(self._mean, self._cov)

    def update(self, det, frame_id):
        x1,y1,x2,y2 = det[:4]
        self._tlwh = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float64)
        self.score = float(det[4])
        self._mean, self._cov = self._kbf.update(self._mean, self._cov, self._to_xyah())
        self.last_frame   = frame_id
        self.tracklet_len += 1
        self.state = 'tracked'

    @property
    def is_confirmed(self):
        return self.tracklet_len >= 2

    @property
    def tlbr(self):
        """[x1,y1,x2,y2] from Kalman 예측 상태"""
        cx,cy,a,h = self._mean[:4]
        w = a * h
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

    @property
    def center(self):
        return int(self._mean[0]), int(self._mean[1])


class ByteTracker:
    """
    ByteTrack 다중 객체 추적기.

    핵심 아이디어 — 2단계 매칭으로 저신뢰도 검출도 활용:
      1차: 활성 트랙  ↔ 고신뢰도 검출 (score ≥ track_thresh)
      2차: 미매칭 트랙 ↔ 저신뢰도 검출 (second_thresh ≤ score < track_thresh)
      3차: 소실 트랙  ↔ 남은 고신뢰도 검출 (재진입 처리)

    매칭 기준: IoU 거리 (1 - IoU), 헝가리안 알고리즘
    """
    def __init__(self, track_thresh=0.6, second_thresh=0.3,
                 match_thresh=0.8, max_lost=30):
        self.track_thresh  = track_thresh   # 고신뢰도 임계값
        self.second_thresh = second_thresh  # 저신뢰도 하한
        self.match_thresh  = match_thresh   # IoU 매칭 임계값
        self.max_lost      = max_lost       # 소실 트랙 유지 프레임 수
        self.tracked  = []   # 활성 트랙
        self.lost     = []   # 소실 트랙 (예측 유지 중)
        self.frame_id = 0
        self._next_id = 1

    # ── IoU 계산 ────────────────────────────────────────────────────
    @staticmethod
    def _iou(b1, b2):
        x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
        x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)

    # ── 매칭 (헝가리안 or 그리디) ───────────────────────────────────
    def _associate(self, tracks, dets, iou_thresh):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = np.array([[1.0 - self._iou(t.tlbr, d[:4])
                          for d in dets] for t in tracks])
        if SCIPY_AVAILABLE:
            ri, ci = linear_sum_assignment(cost)
            pairs   = [(r,c) for r,c in zip(ri,ci) if cost[r,c] <= 1-iou_thresh]
        else:
            # 그리디 폴백 (scipy 미설치 시)
            pairs, used_r, used_c = [], set(), set()
            for c_val,r,c in sorted(
                    [(cost[r,c],r,c) for r in range(len(tracks)) for c in range(len(dets))]):
                if c_val > 1-iou_thresh: break
                if r not in used_r and c not in used_c:
                    pairs.append((r,c)); used_r.add(r); used_c.add(c)
        matched_r={r for r,c in pairs}; matched_c={c for r,c in pairs}
        unmat_t=[i for i in range(len(tracks)) if i not in matched_r]
        unmat_d=[j for j in range(len(dets))   if j not in matched_c]
        return pairs, unmat_t, unmat_d

    # ── 메인 업데이트 ────────────────────────────────────────────────
    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, score]
        반환: 활성 + 확인된(2프레임↑) STrack 목록
        """
        self.frame_id += 1
        high = [d for d in detections if d[4] >= self.track_thresh]
        low  = [d for d in detections if self.second_thresh <= d[4] < self.track_thresh]

        # 모든 트랙 예측
        for t in self.tracked + self.lost:
            t.predict()

        # ── 1차 매칭: 활성 트랙 ↔ 고신뢰도 검출 ──
        m1, ut1, ud1 = self._associate(self.tracked, high, self.match_thresh)
        for ti, di in m1:
            self.tracked[ti].update(high[di], self.frame_id)

        # ── 2차 매칭: 미매칭 활성 트랙 ↔ 저신뢰도 검출 ──
        rem_tracked = [self.tracked[i] for i in ut1]
        m2, ut2, _  = self._associate(rem_tracked, low, 0.5)
        for ti, di in m2:
            rem_tracked[ti].update(low[di], self.frame_id)

        # ── 소실 처리 ──
        for i in ut2:
            rem_tracked[i].state = 'lost'
        newly_lost = [rem_tracked[i] for i in ut2]

        # ── 3차 매칭: 소실 트랙 ↔ 남은 고신뢰도 (재진입) ──
        rem_high = [high[i] for i in ud1]
        m3, _, ud3 = self._associate(self.lost, rem_high, 0.5)
        for ti, di in m3:
            self.lost[ti].update(rem_high[di], self.frame_id)

        # ── 신규 트랙 생성 ──
        new_tracks = [STrack(rem_high[i], self._next_id + k, self.frame_id)
                      for k, i in enumerate(ud3)]
        self._next_id += len(new_tracks)

        # ── 소실 트랙 age 관리 ──
        self.lost = [t for t in self.lost + newly_lost
                     if self.frame_id - t.last_frame <= self.max_lost
                     and t.state == 'lost']

        # ── 활성 트랙 갱신 ──
        self.tracked = ([t for t in self.tracked if t.state == 'tracked']
                        + [t for t in self.lost if t.state == 'tracked']
                        + new_tracks)

        return [t for t in self.tracked if t.is_confirmed]


# ─────────────────────────────────────────────────────────────────────
# [4] LSTM 기반 미래 위치 예측기
# ─────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, out_steps):
            super().__init__()
            self.lstm = nn.LSTM(3, 64, num_layers=2, batch_first=True, dropout=0.1)
            self.fc   = nn.Linear(64, out_steps*3)
            self.out_steps = out_steps
        def forward(self, x):
            h,_ = self.lstm(x)
            return self.fc(h[:,-1,:]).view(-1, self.out_steps, 3)

class LSTMPredictor:
    """온라인 학습 LSTM 궤적 예측기. PyTorch 없으면 선형 외삽으로 대체."""
    SEQ_LEN=30; TRAIN_EVERY=15; EPOCHS=5

    def __init__(self, steps):
        self.steps=steps; self.buf=deque(maxlen=120)
        self.ref=None; self.cnt=0; self.ready=False
        if TORCH_AVAILABLE:
            self.net=_LSTMNet(steps)
            self.opt=optim.Adam(self.net.parameters(), lr=1e-3)
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
        for _ in range(self.EPOCHS):
            self.opt.zero_grad(); loss=self.crit(self.net(X),Y); loss.backward(); self.opt.step()

    def predict(self):
        if len(self.buf)<self.SEQ_LEN: return None
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
    """
    MOTP ≈ Σ(innovation_norm) / Σ(matched_count)
    값이 작을수록 필터 예측이 측정값에 가까움 (추적 정밀도 높음)
    """
    def __init__(self):
        self.total_d=0.0; self.total_c=0; self.frame=0; self._log=[]

    def update(self, innov):
        self.total_d+=innov; self.total_c+=1; self.frame+=1
        self._log.append((self.frame, round(innov*100,3)))

    @property
    def motp_m(self): return self.total_d/self.total_c if self.total_c else 0.0

    def text(self): return f"MOTP:{self.motp_m*100:.2f}cm (N={self.total_c})"

    def save(self, path='motp_log.csv'):
        with open(path,'w',newline='') as f:
            csv.writer(f).writerows([['frame','innovation_cm']]+self._log)
        print(f"📊 MOTP 로그 → {path}")


# ─────────────────────────────────────────────────────────────────────
# [6] 하드웨어 초기화
# ─────────────────────────────────────────────────────────────────────
trt_brain = JetsonTRTEngine('best.engine')

pipeline  = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile   = pipeline.start(rs_config)
align     = rs.align(rs.stream.color)

spatial_filter  = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_fill       = rs.hole_filling_filter()

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
DEPTH_MIN_M = 0.15;  DEPTH_MAX_M = 8.0

print("🚀 AI 추론 엔진 및 RealSense 파이프라인 가동 완료.")

# ─────────────────────────────────────────────────────────────────────
# [7] 인스턴스 생성
# ─────────────────────────────────────────────────────────────────────
# single 모드 — 단일 3D KF
single_kf   = create_tracker(args.model)
# bytetrack 모드 — ByteTracker + 트랙별 3D KF 딕셔너리
byte_tracker = ByteTracker() if args.tracker == 'bytetrack' else None
track_3d_kfs = {}   # {track_id: KFModel3D instance}

# LSTM: single 모드는 1개, bytetrack 모드는 주 트랙에만 적용
lstm_pred = LSTMPredictor(args.predict) if args.predict > 0 else None
motp_eval = MOTPEvaluator()             if args.motp        else None


def get_median_depth(depth_frame, cx, cy, half=2):
    """5×5 영역 유효 깊이 중앙값"""
    vals = [depth_frame.get_distance(cx+dx, cy+dy)
            for dy in range(-half, half+1) for dx in range(-half, half+1)
            if 0 <= cx+dx < 640 and 0 <= cy+dy < 480]
    vals = [v for v in vals if v > 0]
    return float(np.median(vals)) if vals else 0.0


def project_to_pixel(fx, fy, fz):
    """3D 미터 → 2D 픽셀 (화면 범위 밖이면 None)"""
    if fz <= 0: return None
    px = int(fx*intrinsics.fx/fz + intrinsics.ppx)
    py = int(fy*intrinsics.fy/fz + intrinsics.ppy)
    return (px, py) if 0<=px<640 and 0<=py<480 else None


def track_color(tid):
    """트랙 ID → 고유 BGR 색상"""
    np.random.seed(tid * 17 % 256)
    return tuple(int(c) for c in np.random.randint(80, 255, 3))


def draw_info(img, x1, y_top, fx, fy, fz, vx, vy, vz, speed, color=(255,255,0)):
    """위치·속도 텍스트 4줄 오버레이"""
    y = max(y_top, 70)
    cv2.putText(img, f"Offset X:{fx:.2f}m  Y:{fy:.2f}m",
                (x1, y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    cv2.putText(img, f"Alt(Z): {fz:.2f}m",
                (x1, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,0), 1)
    cv2.putText(img, f"Vel Vx:{vx:.2f} Vy:{vy:.2f} Vz:{vz:.2f} m/s",
                (x1, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1)
    cv2.putText(img, f"Speed:{speed:.2f}m/s [{speed*100:.0f}cm/s]",
                (x1, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,140,255), 1)


def send_mavlink(fx, fy, fz):
    """MAVLink LANDING_TARGET 10Hz 송신"""
    global _last_mav_send
    now = time.time()
    if master is None or now - _last_mav_send < 1.0/MAVLINK_SEND_HZ:
        return
    _last_mav_send = now
    master.mav.landing_target_send(
        int(now*1e6), 0, mavutil.mavlink.MAV_FRAME_BODY_NED,
        math.atan2(fx,fz), math.atan2(fy,fz), fz, 0, 0)


# ─────────────────────────────────────────────────────────────────────
# [8] 메인 루프
# ─────────────────────────────────────────────────────────────────────
try:
    while True:
        frames         = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame    = aligned_frames.get_depth_frame()
        color_frame    = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_fill.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())

        # TensorRT 추론 → 8400개 예측 박스
        raw_out = trt_brain.infer(color_image).reshape(5,8400).T

        # ── 검출 박스 변환 (640×640 → 640×480 픽셀) ──────────────────
        all_dets = []
        for pred in raw_out:
            score = float(pred[4])
            if score < 0.3:
                continue
            xc,yc,w,h = pred[:4]
            x1=int(xc-w/2);  y1=int((yc-h/2)*(480/640))
            x2=int(xc+w/2);  y2=int((yc+h/2)*(480/640))
            all_dets.append([x1,y1,x2,y2,score])

        # ══════════════════════════════════════════════════════════════
        # 모드 A: single — 최고 신뢰도 단일 객체
        # ══════════════════════════════════════════════════════════════
        if args.tracker == 'single':
            best = max(all_dets, key=lambda d: d[4]) if all_dets else None

            if best and best[4] >= 0.6:
                x1,y1,x2,y2,score = best
                cx,cy = (x1+x2)//2, (y1+y2)//2

                if 0<=cx<640 and 0<=cy<480:
                    dv = get_median_depth(depth_frame, cx, cy)
                    if DEPTH_MIN_M < dv < DEPTH_MAX_M:
                        rx,ry,rz = rs.rs2_deproject_pixel_to_point(intrinsics,[cx,cy],dv)
                        fx,fy,fz,vx,vy,vz,innov = single_kf.update(rx,ry,rz)
                        speed = math.sqrt(vx**2+vy**2+vz**2)

                        if motp_eval: motp_eval.update(innov)

                        # LSTM
                        futures = None
                        if lstm_pred:
                            lstm_pred.add(fx,fy,fz)
                            futures = lstm_pred.predict()

                        # 시각화
                        cv2.rectangle(color_image,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.circle(color_image,(cx,cy),5,(0,0,255),-1)
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

                        if motp_eval:
                            cv2.putText(color_image,motp_eval.text(),
                                        (5,470),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,255,180),1)

                        send_mavlink(fx,fy,fz)
                        print(f"[{single_kf.model_info}] "
                              f"X:{fx*100:.1f} Y:{fy*100:.1f} Z:{fz*100:.1f}cm | "
                              f"V:{speed*100:.1f}cm/s | innov:{innov*100:.2f}cm"
                              +(f" | {motp_eval.text()}" if motp_eval else ""))
            else:
                single_kf.reset()
                if lstm_pred: lstm_pred.reset()

        # ══════════════════════════════════════════════════════════════
        # 모드 B: bytetrack — 다중 객체 ByteTrack + 트랙별 3D KF
        # ══════════════════════════════════════════════════════════════
        else:
            active = byte_tracker.update(all_dets)

            primary = None   # MAVLink용 최고 신뢰도 트랙 정보

            for track in active:
                tid = track.track_id
                cx,cy = track.center

                if not (0<=cx<640 and 0<=cy<480):
                    continue
                dv = get_median_depth(depth_frame,cx,cy)
                if not (DEPTH_MIN_M < dv < DEPTH_MAX_M):
                    continue

                rx,ry,rz = rs.rs2_deproject_pixel_to_point(intrinsics,[cx,cy],dv)

                # 트랙별 독립 3D 칼만 필터
                if tid not in track_3d_kfs:
                    track_3d_kfs[tid] = create_tracker(args.model)
                fx,fy,fz,vx,vy,vz,innov = track_3d_kfs[tid].update(rx,ry,rz)
                speed = math.sqrt(vx**2+vy**2+vz**2)

                if motp_eval: motp_eval.update(innov)

                # 시각화 — 트랙 ID별 고유 색상
                col  = track_color(tid)
                tlbr = track.tlbr.astype(int)
                cv2.rectangle(color_image,(tlbr[0],tlbr[1]),(tlbr[2],tlbr[3]),col,2)
                cv2.putText(color_image,f"ID{tid} {track.score:.2f}",
                            (tlbr[0],tlbr[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
                draw_info(color_image,tlbr[0],tlbr[1],fx,fy,fz,vx,vy,vz,speed,col)

                # 주 트랙 선택 (최고 신뢰도 → MAVLink)
                if primary is None or track.score > primary[0]:
                    primary = (track.score,tid,fx,fy,fz,vx,vy,vz,innov)

            # 주 트랙 처리 (LSTM + MAVLink)
            if primary:
                _,ptid,fx,fy,fz,vx,vy,vz,innov = primary
                speed = math.sqrt(vx**2+vy**2+vz**2)

                # LSTM — 주 트랙만 적용
                if lstm_pred:
                    lstm_pred.add(fx,fy,fz)
                    futures = lstm_pred.predict()
                    if futures:
                        for k,(px_f,py_f,pz_f) in enumerate(futures):
                            pix=project_to_pixel(px_f,py_f,pz_f)
                            if pix:
                                a=1.0-k/len(futures)
                                cv2.circle(color_image,pix,max(2,int(5*a)),
                                           (int(255*a),int(100*a),255),-1)

                send_mavlink(fx,fy,fz)
                print(f"[ByteTrack|{args.model.upper()}] 활성:{len(active)}개 | "
                      f"주트랙ID:{ptid} score:{primary[0]:.2f} | "
                      f"X:{fx*100:.1f} Y:{fy*100:.1f} Z:{fz*100:.1f}cm | "
                      f"V:{speed*100:.1f}cm/s | innov:{innov*100:.2f}cm"
                      +(f" | {motp_eval.text()}" if motp_eval else ""))
            else:
                if lstm_pred: lstm_pred.reset()

            # 소멸 트랙 3D KF 정리
            active_ids = {t.track_id for t in active}
            for tid in list(track_3d_kfs):
                if tid not in active_ids:
                    track_3d_kfs[tid].reset()
                    del track_3d_kfs[tid]

            # 공통 오버레이
            cv2.putText(color_image,
                        f"ByteTrack | {args.model.upper()} | 활성:{len(active)}",
                        (5,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
            if motp_eval:
                cv2.putText(color_image,motp_eval.text(),
                            (5,470),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,255,180),1)

        # ── 공통 출력 ─────────────────────────────────────────────────
        if out.isOpened():
            out.write(color_image)
        cv2.imshow(f'Precision Landing [{args.tracker}|{args.model.upper()}]', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if motp_eval and motp_eval.total_c > 0:
        motp_eval.save()
        print(f"\n📊 최종 {motp_eval.text()}")
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
