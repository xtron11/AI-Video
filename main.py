import cv2
import mediapipe as mp

# Инициализируем инструменты MediaPipe для отрисовки и поиска позы
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Запускаем захват видео с веб-камеры (0 — это стандартная камера)
cap = cv2.VideoCapture(0)

# Настраиваем модель Holistic (целостный анализ)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Переводим изображение в RGB (MediaPipe работает именно с этим форматом)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Экономим память
        
        # МАГИЯ: Нейросеть ищет точки на теле
        results = holistic.process(image)

        # Перерисовываем обратно в BGR для отображения в OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Рисуем сетку лица (Face Mesh)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        
        # 2. Рисуем скелет тела (Pose)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # 3. Рисуем точки на руках (Left/Right Hands)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Показываем результат в окне
        cv2.imshow('AI HR Assistant - Test', image)

        # Выход из программы при нажатии клавиши 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()