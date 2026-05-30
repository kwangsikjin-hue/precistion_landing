import cv2
import numpy as np
import os
import random

# --- 1. 경로 설정 및 파라미터 ---
PAD_IMG_PATH = "land_mark_3x3m.png"              # 투명 배경의 착륙 패드 이미지
BG_DIR = "backgrounds"                # 다운로드한 아스팔트/시멘트 배경 폴더
OUTPUT_DIR = "dataset"                # 결과물이 저장될 폴더
TOTAL_IMAGES_TO_GENERATE = 1000       # 생성할 총 이미지 장수

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# 이미지 읽기
pad_img = cv2.imread(PAD_IMG_PATH, cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED로 알파채널(투명) 유지
bg_files = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not bg_files:
    print(f"오류: {BG_DIR} 폴더에 아스팔트/시멘트 배경 이미지를 넣어주세요.")
    exit()

print(f"합성 시작... 총 {TOTAL_IMAGES_TO_GENERATE}장의 데이터를 가상 생성합니다.")

for idx in range(TOTAL_IMAGES_TO_GENERATE):
    # 무작위로 배경 이미지 선택 및 읽기
    bg_path = random.choice(bg_files)
    bg = cv2.imread(bg_path)
    
    # 드론 카메라 해상도를 고려해 배경을 640x640 크기로 무작위 크롭 또는 리사이즈
    bg = cv2.resize(bg, (640, 640))
    bg_h, bg_w, _ = bg.shape

    # --- 2. 착륙 패드 무작위 변형 (회전, 스케일, 왜곡) ---
    # 무작위 크기 조절 (배경 대비 15% ~ 45% 크기)
    scale = random.uniform(0.15, 0.45)
    pad_w = int(bg_w * scale)
    pad_h = int(pad_w * (pad_img.shape[0] / pad_img.shape[1]))
    resized_pad = cv2.resize(pad_img, (pad_w, pad_h))

    # 무작위 회전 (0 ~ 360도) 및 원근 왜곡(사선 각도 비행 시뮬레이션)
    angle = random.uniform(0, 360)
    M_rot = cv2.getRotationMatrix2D((pad_w/2, pad_h/2), angle, 1.0)
    rotated_pad = cv2.warpAffine(resized_pad, M_rot, (pad_w, pad_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 원근 왜곡 (Perspective Transform - 멀리서 비스듬히 접근하는 드론 시점 구현)
    pts1 = np.float32([[0,0], [pad_w,0], [0,pad_h], [pad_w,pad_h]])
    dx = random.uniform(0, pad_w * 0.25)
    dy = random.uniform(0, pad_h * 0.25)
    pts2 = np.float32([[dx, dy], [pad_w - dx, dy], [-dx, pad_h - dy], [pad_w + dx, pad_h - dy]])
    M_persp = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_pad = cv2.warpPerspective(rotated_pad, M_persp, (pad_w, pad_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # --- 3. 변형된 패드를 배경에 무작위 위치로 합성 ---
    max_x = bg_w - pad_w
    max_y = bg_h - pad_h
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)

    # 알파 채널을 이용한 마스킹 합성
    alpha_pad = transformed_pad[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_pad

    for c in range(3):
        bg[start_y:start_y+pad_h, start_x:start_x+pad_w, c] = (
            alpha_pad * transformed_pad[:, :, c] +
            alpha_bg * bg[start_y:start_y+pad_h, start_x:start_x+pad_w, c]
        )

    # 무작위 조도 및 노이즈 추가 (햇빛 변동 시뮬레이션)
    brightness = random.randint(-40, 40)
    bg = cv2.convertScaleAbs(bg, alpha=1.0, beta=brightness)

    # --- 4. YOLO 자동 라벨 계산 (Center_X, Center_Y, Width, Height) ---
    # 실제 패드가 합성된 영역의 바운딩 박스 계산
    center_x = (start_x + (pad_w / 2)) / bg_w
    center_y = (start_y + (pad_h / 2)) / bg_h
    yolo_w = pad_w / bg_w
    yolo_h = pad_h / bg_h

    # 파일 저장 (0은 vertiport 클래스 번호 의미)
    file_name = f"synthetic_{idx:04d}"
    cv2.imwrite(f"{OUTPUT_DIR}/images/{file_name}.jpg", bg)
    
    with open(f"{OUTPUT_DIR}/labels/{file_name}.txt", "w") as f:
        f.write(f"0 {center_x:.6f} {center_y:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")

print(f"성공: {OUTPUT_DIR} 폴더에 이미지와 라벨링 파일 생성이 완료되었습니다!")
