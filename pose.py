import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ─────────────────────────────────────────
# БЛОК 1: СКРЕЩЕННЫЕ РУКИ
# Отслеживаем сколько времени кандидат держит руки скрещенными.
# is_actually_crossed — финальный стабильный статус (True только если скрещены > 1 сек)
# crossed_total_seconds — суммарное время скрещивания за всё интервью
# crossed_start — момент когда началось текущее скрещивание (None = не скрещены)
# ─────────────────────────────────────────
last_open_time = time.time()
is_actually_crossed = False
crossed_total_seconds = 0
crossed_start = None

# ─────────────────────────────────────────
# БЛОК 2: КАСАНИЯ ЛИЦА
# Фиксируем сколько раз кандидат подносил руку к лицу — это сигнал тревоги.
# Логика: рука у лица > 0.5 сек → засчитываем касание.
# 3 касания за 30 секунд = один паттерн тревоги (anxiety_score += 1)
# ─────────────────────────────────────────
touch_history = []          # временные метки зафиксированных касаний
nose_touch_start_time = None  # когда рука впервые приблизилась к лицу
hand_in_face_zone = False   # флаг: рука уже засчитана, не дублируем
anxiety_score = 0           # количество паттернов тревоги за интервью

# ─────────────────────────────────────────
# БЛОК 3: АМПЛИТУДА ЖЕСТИКУЛЯЦИИ
# Анализируем насколько активно двигаются руки.
# Храним историю позиций пальцев относительно плеча (не пиксели экрана —
# чтобы смещение корпуса не влияло на результат).
# Считаем стандартное отклонение позиций за последние 2 секунды:
#   < 30  → зажатая жестикуляция (Stiff)
#   30-100 → естественная (Natural)
#   > 100 → избыточная, нервная (Excessive)
# ─────────────────────────────────────────
wrist_history = []          # список {'time', 'side', 'x', 'y'} — позиции пальцев
total_gesture_frames = 0    # сколько кадров попало в анализ
count_stiff = 0             # кадры с зажатой жестикуляцией
count_natural = 0           # кадры с естественной жестикуляцией
count_excessive = 0         # кадры с избыточной жестикуляцией
last_active_time = 0        # когда последний раз видели активное движение рук

cap = cv2.VideoCapture("Video/01_1.mp4")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, img = cap.read()
        if not success: break

        original_img = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Запускаем нейросеть — получаем координаты всех точек тела и рук
        results = holistic.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = img.shape

            # ── Базовые точки тела ──
            # Координаты в пикселях: сырое значение (0-1) умножаем на размер кадра
            l_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            r_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))

            # Ширина плеч в пикселях — используется как адаптивная единица измерения.
            # Все пороги привязаны к ней, поэтому работают при любом расстоянии до камеры
            shoulder_width = np.linalg.norm(np.array(l_shoulder) - np.array(r_shoulder))
            if shoulder_width == 0: shoulder_width = 1  # защита от деления на ноль

            nose    = (int(landmarks[0].x * w),  int(landmarks[0].y * h))
            l_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))
            r_wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))
            l_elbow = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            r_elbow = (int(landmarks[14].x * w), int(landmarks[14].y * h))

            # Расстояния для детектора скрещивания:
            # При скрещенных руках левое запястье оказывается близко к правому локтю и наоборот
            dist1 = np.linalg.norm(np.array(l_wrist) - np.array(r_elbow))
            dist2 = np.linalg.norm(np.array(r_wrist) - np.array(l_elbow))

            # ── Обработка левой руки ──
            # Используем точку 20 (кончик мизинца) — крайняя точка руки,
            # хорошо подходит и для детектора касаний и для амплитуды
            if results.left_hand_landmarks:
                l_finger = results.left_hand_landmarks.landmark[20]
                l_fp = (int(l_finger.x * w), int(l_finger.y * h))
                dist_l_nose = np.linalg.norm(np.array(nose) - np.array(l_fp))
                # Позиция пальца относительно плеча — убирает влияние движения корпуса
                l_rel_x = l_fp[0] - l_shoulder[0]
                l_rel_y = l_fp[1] - l_shoulder[1]
            else:
                dist_l_nose = 9999  # рука не видна — ставим заведомо большое расстояние
                l_rel_x, l_rel_y = None, None

            # ── Обработка правой руки ──
            if results.right_hand_landmarks:
                r_finger = results.right_hand_landmarks.landmark[20]
                r_fp = (int(r_finger.x * w), int(r_finger.y * h))
                dist_r_nose = np.linalg.norm(np.array(nose) - np.array(r_fp))
                r_rel_x = r_fp[0] - r_shoulder[0]
                r_rel_y = r_fp[1] - r_shoulder[1]
            else:
                dist_r_nose = 9999
                r_rel_x, r_rel_y = None, None

            # ── Сбор истории движений ──
            # Каждая рука пишется независимо — если одна не видна, вторая всё равно пишется
            if l_rel_x is not None:
                wrist_history.append({'time': time.time(), 'side': 'l', 'x': l_rel_x, 'y': l_rel_y})
            if r_rel_x is not None:
                wrist_history.append({'time': time.time(), 'side': 'r', 'x': r_rel_x, 'y': r_rel_y})

            now = time.time()
            # Скользящее окно 2 секунды — старые данные автоматически вытесняются
            wrist_history = [p for p in wrist_history if now - p['time'] < 2]

            # Разделяем историю по рукам для независимого подсчёта
            l_data = [p for p in wrist_history if p['side'] == 'l']
            r_data = [p for p in wrist_history if p['side'] == 'r']

            # Стандартное отклонение = насколько сильно разбросаны позиции за 2 сек.
            # Суммируем X и Y отдельно — движение только вбок тоже считается жестикуляцией
            amps = []
            if len(l_data) > 5:
                amps.append(np.std([p['x'] for p in l_data]) + np.std([p['y'] for p in l_data]))
            if len(r_data) > 5:
                amps.append(np.std([p['x'] for p in r_data]) + np.std([p['y'] for p in r_data]))

            # ── Классификация жестикуляции ──
            if amps:
                total_gesture_frames += 1
                avg_amp = sum(amps) / len(amps)

                if avg_amp >= 30:
                    last_active_time = time.time()  # обновляем время последней активности

                if avg_amp > 100:
                    count_excessive += 1
                    gesture_status = "Gestures: Excessive"
                    gesture_color = (0, 255, 255)
                elif avg_amp >= 30:
                    count_natural += 1
                    gesture_status = "Gestures: Natural"
                    gesture_color = (0, 255, 0)
                else:
                    # Руки видны но не двигаются — проверяем льготный период.
                    # 4 секунды после активного жеста считаем естественной паузой
                    if time.time() - last_active_time < 4.0:
                        count_natural += 1
                        gesture_status = "Gestures: Natural (Resting)"
                        gesture_color = (0, 255, 0)
                    else:
                        count_stiff += 1
                        gesture_status = "Gestures: Stiff"
                        gesture_color = (0, 0, 255)

                cv2.putText(img, gesture_status, (50, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
                cv2.putText(img, f"Amp: {int(avg_amp)}", (50, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            else:
                # Руки полностью скрыты (под столом или вне кадра).
                # После льготного периода засчитываем как зажатость —
                # кандидат намеренно убрал руки и не жестикулирует
                if time.time() - last_active_time > 4.0:
                    total_gesture_frames += 1
                    count_stiff += 1
                    gesture_status = "Gestures: Stiff (Hidden)"
                    gesture_color = (0, 0, 255)
                else:
                    gesture_status = "Gestures: Natural (Resting)"
                    gesture_color = (0, 255, 0)

            # Статистика жестикуляции за всё время — выводим в правом верхнем углу
            if total_gesture_frames > 0:
                perc_stiff    = (count_stiff    / total_gesture_frames) * 100
                perc_natural  = (count_natural  / total_gesture_frames) * 100
                perc_excessive= (count_excessive/ total_gesture_frames) * 100

                cv2.putText(img, f"Stiff: {int(perc_stiff)}%",     (w-200, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),   2)
                cv2.putText(img, f"Natural: {int(perc_natural)}%",  (w-200, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),   2)
                cv2.putText(img, f"Excessive: {int(perc_excessive)}%",(w-200,110),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # ── Детектор скрещивания рук ──
            # Порог — 60% от ширины плеч. Если запястье пересекло эту границу
            # И рука реально видна (visibility > 0.5) — считаем что рука скрещена
            adaptive_threshold = shoulder_width * 0.6
            check_l = landmarks[15].visibility > 0.5 and dist1 < adaptive_threshold
            check_r = landmarks[16].visibility > 0.5 and dist2 < adaptive_threshold
            currently_crossed = check_l or check_r

            if not currently_crossed:
                last_open_time = time.time()
                is_actually_crossed = False
                # Руки раскрылись — сохраняем время которое они были скрещены
                if crossed_start is not None:
                    crossed_total_seconds += time.time() - crossed_start
                    crossed_start = None
            elif (time.time() - last_open_time) > 1.0:
                # Скрещены дольше 1 секунды — это не случайное движение
                is_actually_crossed = True
                if crossed_start is None:
                    crossed_start = time.time()  # запускаем таймер один раз

            # ── Детектор касаний лица ──
            # Берём ближайший палец из обеих рук и сравниваем с зоной вокруг носа.
            # Зона = 35% от ширины плеч — покрывает всё лицо адаптивно
            min_dist_nose = min(dist_l_nose, dist_r_nose)
            face_threshold = shoulder_width * 0.35

            if min_dist_nose < face_threshold:
                if nose_touch_start_time is None:
                    nose_touch_start_time = time.time()
                # Засчитываем только если рука держится у лица > 0.5 сек —
                # фильтрует случайные пролёты руки мимо лица
                if (time.time() - nose_touch_start_time) > 0.5 and not hand_in_face_zone:
                    touch_history.append(time.time())
                    hand_in_face_zone = True
            else:
                nose_touch_start_time = None
                hand_in_face_zone = False

            # Скользящее окно 30 секунд для паттерна касаний.
            # 3 касания за 30 секунд = явный паттерн тревоги
            now = time.time()
            touch_history = [t for t in touch_history if now - t < 30]
            if len(touch_history) >= 3:
                anxiety_score += 1
                touch_history.clear()  # сбрасываем серию, начинаем считать заново

            # ── Визуализация текущего состояния ──
            status = "Arms Crossed" if is_actually_crossed else "Open Pose"
            color  = (0, 0, 255)    if is_actually_crossed else (0, 255, 0)
            cv2.putText(img, status,                          (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1,   color,          2)
            cv2.putText(img, f"Anxiety Score: {anxiety_score}",(50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),  2)
            cv2.putText(img, f"Pattern: {len(touch_history)}/3",(50, 130),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1)

            # Отладочный круг — показывает зону детектора касаний вокруг носа
            cv2.circle(img, nose, int(face_threshold), (255, 0, 255), 2)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Result", img)
        cv2.imshow("Original", original_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ═══════════════════════════════════════════
# ИТОГОВЫЙ ОТЧЁТ — формируется после окончания видео
# ═══════════════════════════════════════════

# Индикатор 1 — Жестикуляция (самый важный сигнал)
if total_gesture_frames > 0:
    if perc_natural > 60:
        gest_label = f"Жесты уверенные: {int(perc_natural)}%"
    elif perc_stiff > 50:
        gest_label = f"Жесты пассивные: {int(perc_stiff)}%"
    elif perc_excessive > 40:
        gest_label = f"Жесты чрезмерные: {int(perc_excessive)}%"
    else:
        gest_label = (f"Жесты смешанные:\n"
                      f"  Уверенные: {int(perc_natural)}%\n"
                      f"  Пассивные: {int(perc_stiff)}%\n"
                      f"  Чрезмерные: {int(perc_excessive)}%")
else:
    gest_label = "Жесты: нет данных"

# Индикатор 2 — Касания лица (сигнал тревоги)
if anxiety_score <= 1:
    face_label = "Лицо: Спокойное"
elif anxiety_score <= 3:
    face_label = f"Лицо: Лёгкая тревога"
else:
    face_label = f"Лицо: Встревоженное"

# Индикатор 3 — Скрещенные руки (сигнал закрытости)
# Если сейчас скрещены — добавляем текущий незакрытый отрезок
current_crossed = crossed_total_seconds
if crossed_start is not None:
    current_crossed += time.time() - crossed_start

if current_crossed < 10:
    arms_label = f"Поза: Открытая ({int(current_crossed)} сек скрещено)"
elif current_crossed < 30:
    arms_label = f"Поза: Смешанная ({int(current_crossed)} сек скрещено)"
else:
    arms_label = f"Поза: Закрытая ({int(current_crossed)} сек скрещено)"

print("\n========= ИТОГ ИНТЕРВЬЮ =========")
print(gest_label)
print(face_label)
print(arms_label)
print("=================================")

cap.release()
cv2.destroyAllWindows()