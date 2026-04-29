import time

class AnxietyAnalyzer:
    def __init__(self, threshold_ratio=0.35, min_duration=0.5,
                 window=30, pattern_count=3):
        self.threshold_ratio = threshold_ratio
        self.min_duration = min_duration
        self.window = window
        self.pattern_count = pattern_count

        self.touch_history = []
        self.touch_start = None
        self.in_zone = False
        self.score = 0

    def update(self, dist_l_nose, dist_r_nose, shoulder_width):
        threshold = shoulder_width * self.threshold_ratio
        min_dist = min(dist_l_nose, dist_r_nose)
        now = time.time()

        if min_dist < threshold:
            if self.touch_start is None:
                self.touch_start = now
            if now - self.touch_start > self.min_duration and not self.in_zone:
                self.touch_history.append(now)
                self.in_zone = True
        else:
            self.touch_start = None
            self.in_zone = False

        self.touch_history = [t for t in self.touch_history if now - t < self.window]
        if len(self.touch_history) >= self.pattern_count:
            self.score += 1
            self.touch_history.clear()

        return self.score, len(self.touch_history)

    def get_report(self):
        if self.score <= 1:
            return "Лицо: Спокойное"
        elif self.score <= 3:
            return "Лицо: Лёгкая тревога"
        return "Лицо: Встревоженное"