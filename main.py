import time
import json
import os
from multiprocessing import Pool, cpu_count

import cv2
import mediapipe as mp
import numpy as np

from config import STEP
from analyzers.gaze_analyzer import GazeAnalyzer
from analyzers.pose_analyzer import PoseAnalyzer
from analyzers.gesture_analyzer import GestureAnalyzer
from analyzers.anxiety_analyzer import AnxietyAnalyzer

def process_video_task(video_path):
    print(f" >>> Воркер начал работу над: {video_path}")

    start_video_time = time.time() # Засекаем время внутри функции
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    gaze    = GazeAnalyzer()
    pose    = PoseAnalyzer()
    gesture = GestureAnalyzer()
    anxiety = AnxietyAnalyzer()

    mp_holistic = mp.solutions.holistic
    target_height = 480
    current_time = 0.0

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5,
                              refine_face_landmarks=True) as holistic:
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % STEP != 0:
                continue

            current_time = frame_idx / fps

            h, w = frame.shape[:2]
            ratio = target_height / h
            small_frame = cv2.resize(frame, (int(w * ratio), target_height))
            h, w = small_frame.shape[:2]

            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                l_sh = np.array([lm[11].x * w, lm[11].y * h])
                r_sh = np.array([lm[12].x * w, lm[12].y * h])
                sw = np.linalg.norm(l_sh - r_sh) or 1

                pose.update(lm, sw, current_time)

                l_rel = None
                r_rel = None

                if results.left_hand_landmarks:
                    l_finger = results.left_hand_landmarks.landmark[20]
                    l_rel = (l_finger.x - lm[11].x, l_finger.y - lm[11].y)

                if results.right_hand_landmarks:
                    r_finger = results.right_hand_landmarks.landmark[20]
                    r_rel = (r_finger.x - lm[12].x, r_finger.y - lm[12].y)

                gesture.update(current_time, l_rel, r_rel)

                nose = np.array([lm[0].x * w, lm[0].y * h])
                dist_l = np.linalg.norm(nose - np.array([results.left_hand_landmarks.landmark[20].x * w,
                                                          results.left_hand_landmarks.landmark[20].y * h])) \
                         if results.left_hand_landmarks else 9999
                dist_r = np.linalg.norm(nose - np.array([results.right_hand_landmarks.landmark[20].x * w,
                                                          results.right_hand_landmarks.landmark[20].y * h])) \
                         if results.right_hand_landmarks else 9999
                anxiety.update(dist_l, dist_r, sw, current_time)

            if results.face_landmarks:
                gaze.update(results.face_landmarks.landmark, current_time, STEP)

    cap.release()

    result_data = {
        "video": os.path.basename(video_path),
        "processing_time": f"{time.time() - start_video_time:.2f} сек",
        "gaze": gaze.get_report(),
        "gestures": gesture.get_report(),
        "anxiety": anxiety.get_report(),
        "pose": pose.get_report(),
    }
    return result_data

if __name__ == "__main__":
    video_folder = "./uploads"
    output_folder = "./reports"  # Папка для результатов
    
    # Создаем папку для отчетов, если её еще нет
    os.makedirs(output_folder, exist_ok=True)

    videos = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]
    total_videos = len(videos)
    
    if total_videos == 0:
        print("В папке uploads нет видео!")
        exit()

    num_workers = min(3, max(1, cpu_count() // 2))
    print(f"--- ЗАПУСК СИСТЕМЫ ---")
    print(f"Найдено видео: {total_videos}")
    print(f"Используем ядер: {num_workers}")
    print("-" * 30)
    
    start_time = time.time()

    # Запускаем пул воркеров
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_video_task, videos), 1):
            video_name = result.get("video", "Unknown")
            print(f"[{i}/{total_videos}] Завершена обработка: {video_name}")
            
            # Убираем расширение .mp4 из имени файла и добавляем .json
            base_name = os.path.splitext(video_name)[0]  # Например: "interview_candidate_1"
            json_filename = f"{base_name}_report.json"    # "interview_candidate_1_report.json"
            json_path = os.path.join(output_folder, json_filename)
            
            # Сохраняем индивидуальный JSON сразу по окончании обработки видео
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                
            print(f" -> Отчет успешно сохранен в: {json_path}")

    end_time = time.time()
    print("-" * 30)
    print(f"ВСЕГО ОБРАБОТАНО: {total_videos}")
    print(f"ОБЩЕЕ ВРЕМЯ: {end_time - start_time:.2f} сек")