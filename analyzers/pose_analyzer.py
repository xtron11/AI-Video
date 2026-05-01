import numpy as np

class PoseAnalyzer:
    def __init__(self, threshold_ratio=0.3):
        self.threshold_ratio = threshold_ratio
        self.last_open_time = 0.0
        self.is_crossed = False
        self.crossed_total = 0.0
        self.crossed_start = None

    def update(self, landmarks, shoulder_width, current_time):
        lm = landmarks
        l_wrist = np.array([lm[15].x, lm[15].y])
        r_wrist = np.array([lm[16].x, lm[16].y])
        l_elbow = np.array([lm[13].x, lm[13].y])
        r_elbow = np.array([lm[14].x, lm[14].y])

        dist1 = np.linalg.norm(l_wrist - r_elbow) * shoulder_width
        dist2 = np.linalg.norm(r_wrist - l_elbow) * shoulder_width
        threshold = shoulder_width * self.threshold_ratio

        # Классическое скрещивание — запястье у чужого локтя
        cross_check = (
            (lm[15].visibility > 0.5 and dist1 < threshold) or
            (lm[16].visibility > 0.5 and dist2 < threshold)
        )

        # Расстояние между запястьями
        wrist_dist = np.linalg.norm(l_wrist - r_wrist) * shoulder_width

        # Кисти близко друг к другу — порог 50% от ширины плеч
        wrists_close = wrist_dist < shoulder_width * 0.15

        currently_crossed = cross_check or wrists_close

        if not currently_crossed:
            self.last_open_time = current_time
            self.is_crossed = False
            if self.crossed_start is not None:
                self.crossed_total += current_time - self.crossed_start
                self.crossed_start = None
        elif current_time - self.last_open_time > 2.0:
            self.is_crossed = True
            if self.crossed_start is None:
                self.crossed_start = current_time
        return self.is_crossed
    
    # нужен для отображения времени на экране
    def get_total_crossed(self, current_time):
        total = self.crossed_total
        if self.crossed_start is not None:
            total += current_time - self.crossed_start
        return total

    def get_report(self, current_time):
        t = self.get_total_crossed(current_time)
        if t < 10:
            return f"Поза: Открытая ({int(t)} сек скрещено)"
        elif t < 30:
            return f"Поза: Смешанная ({int(t)} сек скрещено)"
        return f"Поза: Закрытая ({int(t)} сек скрещено)"