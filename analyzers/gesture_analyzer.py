import time
import numpy as np
from config import (GESTURE_WINDOW, GESTURE_STIFF_MAX,
                    GESTURE_EXCESSIVE_MIN, GESTURE_REST_GRACE)

class GestureAnalyzer:
    def __init__(self):
        self.wrist_history = []
        self.total_frames = 0
        self.count_stiff = 0
        self.count_natural = 0
        self.count_excessive = 0
        self.last_active_time = 0

    def update(self, l_rel=None, r_rel=None):
        """
        l_rel, r_rel — кортежи (x, y) позиции пальца относительно плеча.
        Передавай None если рука не видна.
        """
        now = time.time()

        if l_rel is not None:
            self.wrist_history.append({'time': now, 'side': 'l',
                                       'x': l_rel[0], 'y': l_rel[1]})
        if r_rel is not None:
            self.wrist_history.append({'time': now, 'side': 'r',
                                       'x': r_rel[0], 'y': r_rel[1]})

        # Скользящее окно
        self.wrist_history = [p for p in self.wrist_history
                               if now - p['time'] < GESTURE_WINDOW]

        l_data = [p for p in self.wrist_history if p['side'] == 'l']
        r_data = [p for p in self.wrist_history if p['side'] == 'r']

        amps = []
        if len(l_data) > 5:
            amps.append(np.std([p['x'] for p in l_data]) +
                        np.std([p['y'] for p in l_data]))
        if len(r_data) > 5:
            amps.append(np.std([p['x'] for p in r_data]) +
                        np.std([p['y'] for p in r_data]))

        if amps:
            self.total_frames += 1
            avg_amp = sum(amps) / len(amps)

            if avg_amp >= GESTURE_STIFF_MAX:
                self.last_active_time = now

            if avg_amp > GESTURE_EXCESSIVE_MIN:
                self.count_excessive += 1
                return "Excessive", avg_amp
            elif avg_amp >= GESTURE_STIFF_MAX:
                self.count_natural += 1
                return "Natural", avg_amp
            else:
                # Льготный период после активного жеста
                if now - self.last_active_time < GESTURE_REST_GRACE:
                    self.count_natural += 1
                    return "Natural (Resting)", avg_amp
                else:
                    self.count_stiff += 1
                    return "Stiff", avg_amp
        else:
            # Руки не видны
            if now - self.last_active_time > GESTURE_REST_GRACE:
                self.total_frames += 1
                self.count_stiff += 1
                return "Stiff (Hidden)", 0
            return "Natural (Resting)", 0

    def get_percentages(self):
        if self.total_frames == 0:
            return 0, 0, 0
        return (
            self.count_stiff    / self.total_frames * 100,
            self.count_natural  / self.total_frames * 100,
            self.count_excessive/ self.total_frames * 100,
        )

    def get_report(self):
        if self.total_frames == 0:
            return "Жесты: нет данных"
        stiff, natural, excessive = self.get_percentages()
        if natural > 60:
            return f"Жесты уверенные: {int(natural)}%"
        elif stiff > 50:
            return f"Жесты пассивные: {int(stiff)}%"
        elif excessive > 40:
            return f"Жесты чрезмерные: {int(excessive)}%"
        return (f"Жесты смешанные:\n"
                f"  Уверенные: {int(natural)}%\n"
                f"  Пассивные: {int(stiff)}%\n"
                f"  Чрезмерные: {int(excessive)}%")