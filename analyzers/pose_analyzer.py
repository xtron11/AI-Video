import time
import numpy as np

class PoseAnalyzer:
    def __init__(self, threshold_ratio=0.6):
        self.threshold_ratio = threshold_ratio
        self.last_open_time = time.time()
        self.is_crossed = False
        self.crossed_total = 0.0
        self.crossed_start = None

    def update(self, landmarks, shoulder_width):
        lm = landmarks
        l_wrist = np.array([lm[15].x, lm[15].y])
        r_wrist = np.array([lm[16].x, lm[16].y])
        l_elbow = np.array([lm[13].x, lm[13].y])
        r_elbow = np.array([lm[14].x, lm[14].y])

        # используем нормализованные координаты
        scale = shoulder_width
        dist1 = np.linalg.norm(l_wrist - r_elbow) * scale
        dist2 = np.linalg.norm(r_wrist - l_elbow) * scale
        threshold = shoulder_width * self.threshold_ratio

        currently_crossed = (
            (lm[15].visibility > 0.5 and dist1 < threshold) or
            (lm[16].visibility > 0.5 and dist2 < threshold)
        )

        now = time.time()
        if not currently_crossed:
            self.last_open_time = now
            self.is_crossed = False
            if self.crossed_start is not None:
                self.crossed_total += now - self.crossed_start
                self.crossed_start = None
        elif now - self.last_open_time > 1.0:
            self.is_crossed = True
            if self.crossed_start is None:
                self.crossed_start = now

        return self.is_crossed

    def get_total_crossed(self):
        total = self.crossed_total
        if self.crossed_start:
            total += time.time() - self.crossed_start
        return total

    def get_report(self):
        t = self.get_total_crossed()
        if t < 10:
            return f"Поза: Открытая ({int(t)} сек скрещено)"
        elif t < 30:
            return f"Поза: Смешанная ({int(t)} сек скрещено)"
        return f"Поза: Закрытая ({int(t)} сек скрещено)"