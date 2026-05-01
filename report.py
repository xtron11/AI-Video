def print_report(gaze, gesture, anxiety, pose, current_time):
    print("\n========= ИТОГ ИНТЕРВЬЮ =========")
    print(gaze.get_report(current_time))
    print(gesture.get_report())
    print(anxiety.get_report())
    print(pose.get_report(current_time))
    print("=================================")