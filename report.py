def print_report(gaze, gesture, anxiety, pose):
    print("\n========= ИТОГ ИНТЕРВЬЮ =========")
    print(gaze.get_report())
    print(gesture.get_report())
    print(anxiety.get_report())
    print(pose.get_report())
    print("=================================")