import cv2
import mediapipe as mp
import numpy as np
import time

# 顏色設定 (BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
LIGHT_BLUE = (255, 255, 0)
WHITE = (255, 255, 255)

# 載入圖片
naruto_img = cv2.imread('naruto.png')
rasengan_img = cv2.imread('rasangan.png')
if naruto_img is None or rasengan_img is None:
    raise Exception("請確認 naruto.png 與 rasangan.png 是否存在。")
naruto_h, naruto_w = naruto_img.shape[:2]

cap = cv2.VideoCapture(0)

# Mediapipe 初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection()

def rotate(image, angle, center=None, scale=1.0):
    """旋轉影像"""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))

def get_landmark_positions(hand_landmarks, img_w, img_h):
    """將 mediapipe landmark 轉換成絕對座標列表"""
    return [(int(lm.x * img_w), int(lm.y * img_h)) for lm in hand_landmarks.landmark]

def overlay_image(background, overlay, top_left):
    """
    將 overlay 影像依據遮罩覆蓋到 background 上
    """
    x, y = top_left
    h, w = overlay.shape[:2]
    # 產生遮罩 (以灰階及閥值化產生透明部分)
    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    # 取 ROI 區域
    roi = background[y:y+h, x:x+w]
    bg = cv2.bitwise_and(roi, roi, mask=mask)
    fg = cv2.bitwise_and(overlay, overlay, mask=mask_inv)
    combined = cv2.add(bg, fg)
    background[y:y+h, x:x+w] = combined

def overlay_rasengan_on_hand(hand_landmarks, frame, overlay_img):
    """依據手部 landmark 覆蓋並旋轉 rasengan 特效"""
    img_h, img_w = frame.shape[:2]
    positions = get_landmark_positions(hand_landmarks, img_w, img_h)
    if len(positions) < 21:
        return

    # 根據 landmark 4 與 16 計算距離，作為尺寸依據
    p4 = positions[4]
    p16 = positions[16]
    dist = int(np.sqrt((p4[0] - p16[0])**2 + (p4[1] - p16[1])**2)) // 2
    if dist < 50:
        return

    # 選擇 landmark 4, 20, 0, 12 作為中心點計算依據
    indices = [4, 20, 0, 12]
    center_x = sum(positions[i][0] for i in indices) // len(indices)
    center_y = sum(positions[i][1] for i in indices) // len(indices)
    # 根據手型做微調
    if positions[4][0] - positions[20][0] > 0:
        center_x += 15
    else:
        center_x -= 15
    center_y -= 50

    # 根據距離計算 overlay 尺寸與左上角座標
    overlay_size = dist * 2
    # 調整 overlay 尺寸
    overlay_resized = cv2.resize(overlay_img, (overlay_size, overlay_size))
    # 計算旋轉角度 (例如每秒 30 度)
    current_angle = (time.time() * 30) % 360
    rotated_overlay = rotate(overlay_resized, current_angle)
    top_left = (center_x - dist, center_y - dist)
    # 確認不超出邊界
    if top_left[0] < 0 or top_left[1] < 0 or top_left[0] + overlay_size > img_w or top_left[1] + overlay_size > img_h:
        return
    overlay_image(frame, rotated_overlay, top_left)

def overlay_naruto_on_face(bbox, frame, overlay_img):
    """依據臉部偵測的 bounding box 覆蓋 naruto 特效"""
    img_h, img_w = frame.shape[:2]
    # 取得絕對座標與寬高
    xmin = int(bbox.xmin * img_w)
    ymin = int(bbox.ymin * img_h)
    width = int(bbox.width * img_w)
    height = int(bbox.height * img_h)
    center_x = xmin + width // 2
    center_y = ymin + height // 2

    # 根據原始邏輯調整臉部範圍 (可根據需要調整係數)
    adj_width = int(width * 1.75)
    adj_height_top = int(height * 2.5)
    adj_height_bottom = int(height * 1.25)
    xmin_new = max(center_x - adj_width // 2, 0)
    xmax_new = min(center_x + adj_width // 2, img_w)
    ymin_new = max(center_y - adj_height_top // 2, 0)
    ymax_new = min(center_y + adj_height_bottom // 2, img_h)

    overlay_resized = cv2.resize(overlay_img, (xmax_new - xmin_new, ymax_new - ymin_new))
    overlay_image(frame, overlay_resized, (xmin_new, ymin_new))
    # 繪製邊界
    # cv2.rectangle(frame, (xmin_new, ymin_new), (xmax_new, ymax_new), GREEN, 3)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frameRGB)
    face_results = face_detection.process(frameRGB)

    # 處理手部偵測
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            overlay_rasengan_on_hand(hand_landmarks, frame, rasengan_img)
            # 如需顯示 landmark 可啟用下行
            # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 處理臉部偵測
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            overlay_naruto_on_face(bbox, frame, naruto_img)

    # 根據需要縮放影像
    frame = cv2.resize(frame, None, fx=1.3, fy=1.3)
    cv2.imshow('Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
