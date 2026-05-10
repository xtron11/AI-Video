import cv2
import mediapipe as mp
import numpy as np
import time
import psutil
import os

from config import VIDEO_PATH, STEP
from analyzers.gaze_analyzer import GazeAnalyzer
from analyzers.pose_analyzer import PoseAnalyzer
from analyzers.gesture_analyzer import GestureAnalyzer
from analyzers.anxiety_analyzer import AnxietyAnalyzer
from report import print_report

# Инициализация
mp_holistic = mp.solutions.holistic
gaze = GazeAnalyzer()
pose = PoseAnalyzer()
gesture = GestureAnalyzer()
anxiety = AnxietyAnalyzer()

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Метрики
process = psutil.Process(os.getpid())
start_wall_time = time.time()
start_cpu_time = process.cpu_times().user

# Целевая высота для Resize
target_height = 480

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                          refine_face_landmarks=True) as holistic:
    
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_idx += 1
        current_time = frame_idx / fps

        # Пропускаем кадры через STEP
        if frame_idx % STEP == 0:
            # 1. Resize (уменьшаем нагрузку на нейронку)
            h, w = frame.shape[:2]
            ratio = target_height / h
            new_w = int(w * ratio)
            small_frame = cv2.resize(frame, (new_w, target_height))

            # 2. Перевод в RGB
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # 3. Инференс (самая тяжелая часть)
            pose_results = holistic.process(rgb)

            # 4. Обработка результатов (без отрисовки текста)
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                
                # Используем размеры уменьшенного кадра
                l_sh_px = np.array([lm[11].x * new_w, lm[11].y * target_height])
                r_sh_px = np.array([lm[12].x * new_w, lm[12].y * target_height])
                sw_px = np.linalg.norm(l_sh_px - r_sh_px) or 1

                # Обновляем позу
                pose.update(lm, sw_px, current_time)
                
                # Жестикуляция (нормализованные координаты)
                l_rel = None
                if pose_results.left_hand_landmarks:
                    l_finger = pose_results.left_hand_landmarks.landmark[20]
                    l_rel = (l_finger.x - lm[11].x, l_finger.y - lm[11].y)

                r_rel = None
                if pose_results.right_hand_landmarks:
                    r_finger = pose_results.right_hand_landmarks.landmark[20]
                    r_rel = (r_finger.x - lm[12].x, r_finger.y - lm[12].y)

                gesture.update(current_time, l_rel, r_rel, STEP)

                # Тревога (касания лица)
                nose_px = np.array([lm[0].x * new_w, lm[0].y * target_height])
                dist_l = np.linalg.norm(nose_px - np.array([l_finger.x * new_w, l_finger.y * target_height])) \
                         if pose_results.left_hand_landmarks else 9999
                dist_r = np.linalg.norm(nose_px - np.array([r_finger.x * new_w, r_finger.y * target_height])) \
                         if pose_results.right_hand_landmarks else 9999
                anxiety.update(dist_l, dist_r, sw_px, current_time)

            # Взгляд
            if pose_results.face_landmarks:
                gaze.update(pose_results.face_landmarks.landmark, current_time, STEP)

cap.release()

# Итоговые метрики
end_wall_time = time.time()
end_cpu_time = process.cpu_times().user

total_wall = end_wall_time - start_wall_time
total_cpu = end_cpu_time - start_cpu_time

print_report(gaze, gesture, anxiety, pose, current_time)

print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТА (NEW CODE + HEADLESS) ===")
print(f"Реальное время выполнения: {total_wall:.2f} сек")
print(f"Затрачено процессорного времени: {total_cpu:.2f} сек")
print(f"Эффективность (Video Time / Exec Time): {current_time / total_wall:.2f}x")