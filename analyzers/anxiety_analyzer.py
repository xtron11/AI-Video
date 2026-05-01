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

    def update(self, dist_l_nose, dist_r_nose, shoulder_width, current_time):
        threshold = shoulder_width * self.threshold_ratio
        min_dist = min(dist_l_nose, dist_r_nose)

        # 1. Проверка касания (порог расстояния)
        if min_dist < threshold:
            if self.touch_start is None:
                # Фиксируем время начала касания по шкале видео
                self.touch_start = current_time
            
            # Проверяем, длится ли касание дольше минимального (например, 0.5 сек)
            # И не засчитали ли мы это конкретное касание уже (in_zone)
            if (current_time - self.touch_start) > self.min_duration and not self.in_zone:
                self.touch_history.append(current_time)
                self.in_zone = True
        else:
            # Сброс, если рука отошла от лица
            self.touch_start = None
            self.in_zone = False

        # 2. Очистка истории от старых касаний (вне окна, например, 30 сек)
        self.touch_history = [t for t in self.touch_history if current_time - t < self.window]
        
        # 3. Если за окно времени набралось нужное кол-во касаний (pattern_count)
        if len(self.touch_history) >= self.pattern_count:
            self.score += 1
            # Очищаем историю, чтобы начать считать новый паттерн
            self.touch_history.clear()

        return self.score, len(self.touch_history)

    def get_report(self):
        if self.score <= 1:
            return "Лицо: Спокойное"
        elif self.score <= 3:
            return "Лицо: Лёгкая тревога"
        return "Лицо: Встревоженное"