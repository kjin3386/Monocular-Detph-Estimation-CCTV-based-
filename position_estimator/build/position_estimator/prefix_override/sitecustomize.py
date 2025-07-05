import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eojin/ros2_ws/src/position_estimator/install/position_estimator'
