import numpy as np

class GazeAnalyzer:
    def __init__(self):
        self.look_count = 0
        self.total_frames = 0
        self.total_blinks = 0
        self.last_blink_time = 0.0
        self.total_v_duration = 0.0
        
        self.blink_map = {}          # {секунда: количество_морганий}
        self.stiff_episodes = []     # Эпизоды замирания

        self.ear_history = []
        self.EAR_THRESHOLD = None
        self.CALIBRATION_TIME = 3.0   # Калибровка 3 секунды
        self.STIFF_THRESHOLD = 10.0   # Порог замирания (10 секунд без морганий)
        
        # Добавляем переменные счетчиков фреймов (они необходимы для работы детектора)
        self.closed_frames = 0
        self.open_frames = 0
        self.MIN_OPEN_FRAMES = 2

        self.prev_ear = None
        self.EAR_DROP_THRESHOLD = 0.06  # порог резкости падения
    def _get_point(self, lm, idx):
        return np.array([lm[idx].x, lm[idx].y])

    def calculate_ear(self, lm):
        def eye_aspect_ratio(p1, p2, p3, p4, p5, p6):
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h  = np.linalg.norm(p1 - p4)
            return (v1 + v2) / (2.0 * h)

        l_ear = eye_aspect_ratio(self._get_point(lm, 33), self._get_point(lm, 160), self._get_point(lm, 158), self._get_point(lm, 133), self._get_point(lm, 153), self._get_point(lm, 144))
        r_ear = eye_aspect_ratio(self._get_point(lm, 362), self._get_point(lm, 385), self._get_point(lm, 387), self._get_point(lm, 263), self._get_point(lm, 373), self._get_point(lm, 380))
        return (l_ear + r_ear) / 2.0

    def format_time(self, seconds):
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"

    def update(self, lm, current_v_time, STEP):
        self.total_v_duration = current_v_time
        self.total_frames += STEP
        
        # 1. Анализ взгляда
        head_ok = abs((self._get_point(lm, 1) - (self._get_point(lm, 33) + self._get_point(lm, 263)) / 2)[0]) < 0.05
        l_ratio = (lm[468].x - lm[33].x) / (lm[133].x - lm[33].x)
        r_ratio = (lm[473].x - lm[362].x) / (lm[263].x - lm[362].x)
        looking = head_ok and (0.4 < l_ratio < 0.6 and 0.4 < r_ratio < 0.6)

        if looking:
            self.look_count += STEP

        # 2. Калибровка
        ear = self.calculate_ear(lm)
        if self.EAR_THRESHOLD is None:
            self.ear_history.append(ear)
            if current_v_time >= self.CALIBRATION_TIME:
                baseline = np.mean(self.ear_history)
                self.EAR_THRESHOLD = baseline * 0.83
            return
        
        # Проверяем, лежат ли зрачки внутри горизонтальных границ глазных уголков
        left_iris_visible = lm[33].x < lm[468].x < lm[133].x
        right_iris_visible = lm[362].x < lm[473].x < lm[263].x
        
        # Нам достаточно, чтобы был виден хотя бы один зрачок (если голова повернута вбок)
        iris_visible = left_iris_visible or right_iris_visible

        # Если зрачков не видно, мы просто полностью ИГНОРИРУЕМ этот кадр для детектора морганий.
        if iris_visible:
            # Считаем дельту, только если есть предыдущий кадр
            drop = self.prev_ear - ear if self.prev_ear is not None else 0.0
            
            # Условие закрытия: ИЛИ EAR упал ниже порога калибровки, ИЛИ произошел резкий дроп
            is_closing = (ear < self.EAR_THRESHOLD) or (drop > self.EAR_DROP_THRESHOLD)
            
            if is_closing:
                self.closed_frames += 1
                self.open_frames = 0
                
                if self.closed_frames == 1:  # Фиксируем моргание ТОЛЬКО в первый кадр закрытия
                    if self.last_blink_time == 0.0:
                        self.last_blink_time = current_v_time

                    time_since_last = current_v_time - self.last_blink_time
                    if time_since_last > self.STIFF_THRESHOLD:
                        self.stiff_episodes.append({
                            "start": self.format_time(self.last_blink_time),
                            "end": self.format_time(current_v_time),
                            "duration_sec": int(time_since_last)
                        })

                    self.total_blinks += 1
                    self.last_blink_time = current_v_time
                    
                    sec = int(current_v_time)
                    self.blink_map[sec] = self.blink_map.get(sec, 0) + 1
            else:
                # Глаз открыт
                self.open_frames += 1
                if self.open_frames >= self.MIN_OPEN_FRAMES:
                    self.closed_frames = 0
            
            self.prev_ear = ear  # Запоминаем EAR текущего кадра для следующего шага

    def get_report(self):
        if self.total_frames == 0 or self.total_v_duration == 0:
            return {"status": "no_data"}

        self.last_blink_time = self.total_v_duration

        pct = (self.look_count / self.total_frames) * 100
        global_avg_bpm = (self.total_blinks / self.total_v_duration) * 60

        anomalies = []
        window_size = 15

        for start_t in range(0, int(self.total_v_duration) - window_size, 5):
            end_t = start_t + window_size
            blinks_in_window = sum(count for t, count in self.blink_map.items() if start_t <= t < end_t)
            window_bpm = (blinks_in_window / window_size) * 60

            if window_bpm > (global_avg_bpm * 1.5) and window_bpm > 25:
                if anomalies and anomalies[-1]['end_sec'] >= start_t:
                    anomalies[-1]['end_sec'] = end_t
                else:
                    anomalies.append({"start_sec": start_t, "end_sec": end_t})
                    
        return {
            "look_confidence_percent": round(pct, 1),
            "anxiety_episodes": [
                {
                    "start": self.format_time(a['start_sec']),
                    "end": self.format_time(a['end_sec']),
                } for a in anomalies
            ],
            "focus_episodes": self.stiff_episodes
        }