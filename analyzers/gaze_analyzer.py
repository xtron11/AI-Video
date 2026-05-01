import numpy as np

class GazeAnalyzer:
    def __init__(self):
        self.look_count = 0
        self.total_frames = 0 # Будем просто считать кадры для статистики

        self.total_blinks = 0
        self.blink_count_at_start = 0
        self.blink_timestamps = []
        self.closed_frames = 0
        self.open_frames = 0
        self.MIN_OPEN_FRAMES = 2
        self.last_blink_time = 0
        self.peak_bpm = 0

        self.anomalies = []
        self.stiff_episodes = []

        self.current_anomaly_start = None
        self.is_stiff_recorded = False
        self.last_bpm_val = 0

        self.ear_history = []
        self.EAR_THRESHOLD = None
        # Калибровка по времени, а не по кадрам
        self.CALIBRATION_TIME = 3.0 

        self.BPM_THRESHOLD = 30
        self.STIFF_THRESHOLD = 10.0

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

    def update(self, lm, current_v_time):
        self.total_frames += 1 # Для итогового отчета
        
        # 1. Анализ взгляда
        head_ok = abs((self._get_point(lm, 1) - (self._get_point(lm, 33) + self._get_point(lm, 263)) / 2)[0]) < 0.05
        l_ratio = (lm[468].x - lm[33].x) / (lm[133].x - lm[33].x)
        r_ratio = (lm[473].x - lm[362].x) / (lm[263].x - lm[362].x)
        looking = head_ok and (0.4 < l_ratio < 0.6 and 0.4 < r_ratio < 0.6)
        if looking:
            self.look_count += 1

        # 2. Калибровка (используем время)
        ear = self.calculate_ear(lm)
        if self.EAR_THRESHOLD is None:
            self.ear_history.append(ear)
            if current_v_time >= self.CALIBRATION_TIME:
                baseline = np.mean(self.ear_history)
                self.EAR_THRESHOLD = baseline * 0.75
                print(f"[Calibration] baseline: {baseline:.3f} -> threshold: {self.EAR_THRESHOLD:.3f}")
            return looking, 0, "Calibrating...", 0

        # 3. Детектор моргания
        if ear < self.EAR_THRESHOLD:
            self.closed_frames += 1
            self.open_frames = 0
            if self.closed_frames == 1:
                self.total_blinks += 1
                if self.is_stiff_recorded:
                    self.stiff_episodes.append((self.last_blink_time, current_v_time))
                    self.is_stiff_recorded = False
                self.blink_timestamps.append(current_v_time)
                self.last_blink_time = current_v_time
        else:
            self.open_frames += 1
            if self.open_frames >= self.MIN_OPEN_FRAMES:
                self.closed_frames = 0

        # 4. Расчёт BPM
        self.blink_timestamps = [t for t in self.blink_timestamps if current_v_time - t < 60]
        calc_t = min(current_v_time, 60)
        bpm = 0 if current_v_time < 10 else (len(self.blink_timestamps) / calc_t) * 60

        # 5. Логика высокого BPM
        if bpm > self.BPM_THRESHOLD:
            if self.current_anomaly_start is None:
                self.current_anomaly_start = current_v_time
                self.blink_count_at_start = self.total_blinks
                self.peak_bpm = bpm
            else:
                self.peak_bpm = max(self.peak_bpm, bpm)
        else: # BPM упал ниже порога
            if self.current_anomaly_start is not None:
                duration = current_v_time - self.current_anomaly_start
                if duration > 2.0:
                    blinks_in_episode = self.total_blinks - self.blink_count_at_start
                    
                    # Сохраняем пиковое значение, которое мы отслеживали всё это время
                    self.anomalies.append((
                        self.current_anomaly_start, 
                        current_v_time, 
                        self.peak_bpm,  # <--- Вот он, наш пик
                        blinks_in_episode
                    ))
                self.current_anomaly_start = None
                self.peak_bpm = 0 # Сбрасываем пик для следующего раза

        # 6. Логика замирания
        time_since_last_blink = current_v_time - self.last_blink_time
        focus_status = "Normal"
        if self.closed_frames == 0 and time_since_last_blink > self.STIFF_THRESHOLD:
            self.is_stiff_recorded = True
            focus_status = "Deep Focus"

        self.last_bpm_val = bpm
        return looking, bpm, focus_status, time_since_last_blink

    def format_time(self, seconds):
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"

    def get_report(self, current_time):
        if self.total_frames == 0: return "Взгляд: нет данных"

        if self.current_anomaly_start is not None:
            blinks_in_episode = self.total_blinks - self.blink_count_at_start
            self.anomalies.append((self.current_anomaly_start, current_time, self.peak_bpm, blinks_in_episode))
        
        if self.is_stiff_recorded:
            self.stiff_episodes.append((self.last_blink_time, current_time))

        pct = (self.look_count / self.total_frames) * 100
        report = [f"Взгляд: {'Уверенный' if pct > 70 else 'Неуверенный'} ({pct:.1f}% времени)"]

        if self.anomalies:
            report.append("\nУчастки тревожности (высокий BPM):")
            for s, e, peak_b, count in self.anomalies:
                report.append(f"  - [{self.format_time(s)}-{self.format_time(e)}] Морганий: {count} (Пиковый BPM: {int(peak_b)})")

        if self.stiff_episodes:
            report.append("\nУчастки глубокой фокусировки (замирания):")
            for s, e in self.stiff_episodes:
                report.append(f"  - [{self.format_time(s)}-{self.format_time(e)}] Длительность: {int(e - s)} сек.")

        return "\n".join(report)