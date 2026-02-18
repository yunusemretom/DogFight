import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/tom/Documents/Dog_Fight/ros2_ws/install/object_detection'
