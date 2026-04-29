import cv2
import mediapipe as mp
import numpy as np
from config import VIDEO_PATH
from analyzers.gaze_analyzer import GazeAnalyzer
from analyzers.pose_analyzer import PoseAnalyzer
from analyzers.gesture_analyzer import GestureAnalyzer
from analyzers.anxiety_analyzer import AnxietyAnalyzer
from report import print_report

mp_holistic  = mp.solutions.holistic

gaze    = GazeAnalyzer()
pose    = PoseAnalyzer()
gesture = GestureAnalyzer()
anxiety = AnxietyAnalyzer()

cap = cv2.VideoCapture(VIDEO_PATH)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                          refine_face_landmarks=True) as holistic:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        pose_results = holistic.process(rgb)

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            l_sh = np.array([lm[11].x * w, lm[11].y * h])
            r_sh = np.array([lm[12].x * w, lm[12].y * h])
            sw   = np.linalg.norm(l_sh - r_sh) or 1

            pose.update(lm, sw)

            # ── Жестикуляция ──
            if pose_results.left_hand_landmarks:
                l_finger = pose_results.left_hand_landmarks.landmark[20]
                l_rel = (l_finger.x * w - l_sh[0],
                         l_finger.y * h - l_sh[1])
            else:
                l_rel = None

            if pose_results.right_hand_landmarks:
                r_finger = pose_results.right_hand_landmarks.landmark[20]
                r_rel = (r_finger.x * w - r_sh[0],
                         r_finger.y * h - r_sh[1])
            else:
                r_rel = None

            gesture_status, amp = gesture.update(l_rel, r_rel)

            # ── Тревога (касания лица) ──
            nose = np.array([lm[0].x * w, lm[0].y * h])
            dist_l = np.linalg.norm(nose - np.array([l_finger.x * w, l_finger.y * h])) \
                     if pose_results.left_hand_landmarks else 9999
            dist_r = np.linalg.norm(nose - np.array([r_finger.x * w, r_finger.y * h])) \
                     if pose_results.right_hand_landmarks else 9999
            anxiety.update(dist_l, dist_r, sw)

        # ── FaceMesh (взгляд) ──
        if pose_results.face_landmarks:
            # Получаем данные из анализатора
            looking, current_bpm, focus_status, stiff_time = gaze.update(pose_results.face_landmarks.landmark)

            # Цвета для индикации
            # Фокус — синий, Тревога — красный, Норма — зеленый
            color = (0, 255, 0)
            if focus_status == "Deep Focus":
                color = (255, 191, 0) # Голубой
            elif current_bpm > gaze.BPM_THRESHOLD:
                color = (0, 0, 255) # Красный

            # --- Отрисовка на экране ---
            
            # 1. Основной статус (Looking / Not Looking)
            look_text = "LOOKING" if looking else "NOT LOOKING"
            cv2.putText(frame, look_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 2. Текущий BPM
            cv2.putText(frame, f"BPM: {int(current_bpm)}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 3. Таймер без морганий (чтобы ты видел, как тикают те самые 10 секунд)
            stiff_color = (255, 255, 0) if stiff_time > gaze.STIFF_THRESHOLD else (0, 200, 200)
            cv2.putText(frame, f"No Blink: {stiff_time:.1f}s", (30, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, stiff_color, 2)

            # 4. Плашка статуса (появляется только при аномалиях)
            if focus_status == "Deep Focus":
                cv2.rectangle(frame, (25, 130), (200, 160), (255, 191, 0), -1)
                cv2.putText(frame, "STIFF / FOCUS", (40, 152), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        cv2.imshow("Interview Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print_report(gaze, gesture, anxiety, pose)