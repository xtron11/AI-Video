import numpy as np

class PoseAnalyzer:
    def __init__(self, threshold_ratio=0.3):
        self.threshold_ratio = threshold_ratio
        self.pose_episodes = []           # Сюда пишем готовые эпизоды
        self.current_episode_start = None  # Таймкод начала текущей закрытой позы
        self.last_open_time = 0.0         # Время, когда поза последний раз была открытой
        self.is_crossed = False
        self.total_video_duration = 0.0   # Для закрытия последнего эпизода

    def format_time(self, seconds):
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
    
    def _to_seconds(self, time_str):
        m, s = time_str.split(":")
        return int(m) * 60 + int(s)
    
    def update(self, lm, shoulder_width, current_time):
        self.total_video_duration = current_time # Запоминаем последний таймкод
        
        # Векторизация координат
        l_wrist = np.array([lm[15].x, lm[15].y])
        r_wrist = np.array([lm[16].x, lm[16].y])
        l_elbow = np.array([lm[13].x, lm[13].y])
        r_elbow = np.array([lm[14].x, lm[14].y])

        # Расчет дистанций
        dist1 = np.linalg.norm(l_wrist - r_elbow) * shoulder_width
        dist2 = np.linalg.norm(r_wrist - l_elbow) * shoulder_width
        threshold = shoulder_width * self.threshold_ratio

        # Скрещивание — левое запястье у правого локтя ИЛИ правое запястье у левого локтя
        # Проверяем видимость ОБОИХ суставов в паре
        cross_check = (
            (lm[15].visibility > 0.65 and lm[14].visibility > 0.65 and dist1 < threshold) or
            (lm[16].visibility > 0.65 and lm[13].visibility > 0.65 and dist2 < threshold)
        )

        # Расстояние между запястьями
        wrist_dist = np.linalg.norm(l_wrist - r_wrist) * shoulder_width
        
        wrists_visible = lm[15].visibility > 0.65 and lm[16].visibility > 0.65
        wrists_close = wrists_visible and (wrist_dist < shoulder_width * 0.15)

        currently_crossed = cross_check or wrists_close

        if not currently_crossed:
            # Если поза открыта, обновляем таймер "последнего раза, когда видели открытую позу"
            self.last_open_time = current_time
            
            if self.is_crossed:
                episode_end = current_time
                duration = episode_end - self.current_episode_start
                
                if duration >= 3.0:
                    self.pose_episodes.append({
                        "pose": "closed",
                        "start": self.format_time(self.current_episode_start), # Исправлено имя метода
                        "end": self.format_time(episode_end),
                        "duration_sec": round(duration, 1)
                    })
                
                self.is_crossed = False
                self.current_episode_start = None

        else:
            # Если поза закрыта более 2 секунд — считаем это эпизодом
            if current_time - self.last_open_time > 2.0:
                if not self.is_crossed:
                    self.is_crossed = True
                    # Начало — это тот момент, когда открытая поза закончилась
                    self.current_episode_start = self.last_open_time 

        return self.is_crossed

    def get_report(self):
        if not self.pose_episodes:
            return {"status": "no_data"}

        # Склеиваем эпизоды с паузой меньше 3 секунд
        merged = []
        for ep in self.pose_episodes:
            if merged and self._to_seconds(ep["start"]) - self._to_seconds(merged[-1]["end"]) < 3:
                merged[-1]["end"] = ep["end"]
                merged[-1]["duration_sec"] = round(
                    self._to_seconds(merged[-1]["end"]) - self._to_seconds(merged[-1]["start"]), 1
                )
            else:
                merged.append(ep)

        return {"pose_episodes": merged}