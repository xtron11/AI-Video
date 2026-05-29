import numpy as np
from config import (GESTURE_WINDOW, GESTURE_STIFF_MAX,
                    GESTURE_EXCESSIVE_FALLBACK, GESTURE_REST_GRACE)

class GestureAnalyzer:
    def __init__(self):
        self.wrist_history = []
        self.amp_history = []  # история всех амплитуд за интервью

        self.total_frames = 0
        self.count_stiff = 0
        self.count_natural = 0
        self.count_excessive = 0
        self.last_active_time = 0

    def update(self, current_time=0, l_rel=None, r_rel=None, STEP=2):
        """
        l_rel, r_rel — кортежи (x, y) позиции пальца относительно плеча.
        """

        if l_rel is not None:
            self.wrist_history.append({'time': current_time, 'side': 'l',
                                       'x': l_rel[0], 'y': l_rel[1]})
        if r_rel is not None:
            self.wrist_history.append({'time': current_time, 'side': 'r',
                                       'x': r_rel[0], 'y': r_rel[1]})

        # Скользящее окно
        self.wrist_history = [p for p in self.wrist_history
                               if current_time - p['time'] < GESTURE_WINDOW]

        l_data = [p for p in self.wrist_history if p['side'] == 'l']
        r_data = [p for p in self.wrist_history if p['side'] == 'r']

        amps = []
        if len(l_data) > 15:
            amps.append(np.std([p['x'] for p in l_data]) +
                        np.std([p['y'] for p in l_data]))
        if len(r_data) > 15:
            amps.append(np.std([p['x'] for p in r_data]) +
                        np.std([p['y'] for p in r_data]))

        if amps:
            self.total_frames += STEP
            avg_amp = sum(amps) / len(amps)
            self.amp_history.append(avg_amp)

            if avg_amp >= GESTURE_STIFF_MAX:
                self.last_active_time = current_time

            # Динамический порог excessive — выброс для этого конкретного человека
            if len(self.amp_history) > 60:
                mean = np.mean(self.amp_history)
                std  = np.std(self.amp_history)
                dynamic_excessive = mean + 2 * std
            else:
                dynamic_excessive = GESTURE_EXCESSIVE_FALLBACK   # fallback первые 2 секунды

            if avg_amp > dynamic_excessive:
                self.count_excessive += STEP
            elif avg_amp >= GESTURE_STIFF_MAX:
                self.count_natural += STEP
            else:
                if current_time - self.last_active_time < GESTURE_REST_GRACE:
                    self.count_natural += STEP
                else:
                    self.count_stiff += STEP

            return avg_amp

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
            return {"status": "no_data"}
            
        stiff, natural, excessive = self.get_percentages()
        
        # Определяем доминирующий статус для удобства
        if natural > 60:
            status = "confident"
        elif stiff > 50:
            status = "passive"
        elif excessive > 40:
            status = "excessive"
        else:
            status = "mixed"
            
        return {
            "gesture_status": status,
            "percentages": {
                "natural": int(natural),
                "stiff": int(stiff),
                "excessive": int(excessive)
            }
        }