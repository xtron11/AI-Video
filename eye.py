import cv2
import mediapipe as mp
import numpy as np

VIDEO_PATH = "Video/01_1.mp4"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
look_count = 0

def get_point(lm, idx):
    return np.array([lm[idx].x, lm[idx].y])

def head_pose_ok(lm):
    left_eye = get_point(lm, 33)
    right_eye = get_point(lm, 263)
    nose = get_point(lm, 1)

    eye_center = (left_eye + right_eye) / 2
    vec = nose - eye_center
    return abs(vec[0]) < 0.05

def eye_gaze_ok(lm):
    l_outer = get_point(lm, 33)
    l_inner = get_point(lm, 133)
    l_iris = get_point(lm, 468)

    r_inner = get_point(lm, 362)
    r_outer = get_point(lm, 263)
    r_iris = get_point(lm, 473)

    l_ratio = (l_iris[0] - l_outer[0]) / (l_inner[0] - l_outer[0])
    r_ratio = (r_iris[0] - r_inner[0]) / (r_outer[0] - r_inner[0])

    return 0.4 < l_ratio < 0.6 and 0.4 < r_ratio < 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    looking = False

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        head_ok = head_pose_ok(lm)
        eyes_ok = eye_gaze_ok(lm)

        looking = head_ok and eyes_ok

        if looking:
            look_count += 1

        for idx in [33,133,362,263,468,473]:
            p = lm[idx]
            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 2, (0,255,0), -1)

        color = (0,255,0) if looking else (0,0,255)

        cv2.circle(frame, (int(lm[468].x*w), int(lm[468].y*h)), 4, color, -1)
        cv2.circle(frame, (int(lm[473].x*w), int(lm[473].y*h)), 4, color, -1)

        cv2.putText(frame, f"Head: {'OK' if head_ok else 'TURNED'}",
                    (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    text = "LOOKING" if looking else "NOT LOOKING"
    cv2.putText(frame, text, (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Gaze", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"{look_count/frame_count*100:.2f}% frames looking")