#!/usr/bin/env python3

from __future__ import print_function
import threading
import rospy
from std_msgs.msg import String
import sys
from select import select
import termios
import tty


help_msg = """
Control Your Robot with Keyboard!
----------------------------------
Movement keys:
  w    - Forward
  s    - Backward
  a    - Left
  d    - Right
  i    - Up
  k    - Down
  j    - Rotate CCW
  l    - Rotate CW

Commands:
  t    - Takeoff
  g    - Land
  q    - Stop

  h    - Show this help
  Ctrl-C to quit
"""


# 控制处理类
class control_handler: 
    def __init__(self, control_pub):
        self.control_pub = control_pub
    
    def forward(self, cm):
        command = "forward "+(str(cm))
        self.control_pub.publish(command)
    
    def back(self, cm):
        command = "back "+(str(cm))
        self.control_pub.publish(command)
    
    def up(self, cm):
        command = "up "+(str(cm))
        self.control_pub.publish(command)
    
    def down(self, cm):
        command = "down "+(str(cm))
        self.control_pub.publish(command)
    
    def right(self, cm):
        command = "right "+(str(cm))
        self.control_pub.publish(command)
    
    def left(self, cm):
        command = "left "+(str(cm))
        self.control_pub.publish(command)

    def cw(self, cm):
        command = "cw "+(str(cm))
        self.control_pub.publish(command)

    def ccw(self, cm):
        command = "ccw "+(str(cm))
        self.control_pub.publish(command)

    def takeoff(self):
        command = "takeoff"
        self.control_pub.publish(command)
        print ("ready")
        
    def mon(self):
        command = "mon"
        self.control_pub.publish(command)
        print ("mon")

    def land(self):
        command = "land"
        self.control_pub.publish(command)

# 键盘输入处理
def getKey(settings, timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def saveTerminalSettings():
    return termios.tcgetattr(sys.stdin)

def restoreTerminalSettings(old_settings):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

# 发布线程
class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        self.publisher = rospy.Publisher('sdk_cmd', String, queue_size=1)
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def stop(self):
        self.done = True
        self.join()

    def run(self):
        while not self.done:
            rospy.sleep(self.timeout)

# 主程序
if __name__=="__main__":
    settings = saveTerminalSettings()

    rospy.init_node('keyboard_control_handler')

    pub_thread = PublishThread(0.1)
    controller = control_handler(pub_thread.publisher)

    try:
        pub_thread.wait_for_subscribers()
        print(help_msg)
        while True:
            key = getKey(settings, 0.1)

            if key == 'w':
                controller.forward(20)
            elif key == 's':
                controller.back(20)
            elif key == 'a':
                controller.left(20)
            elif key == 'd':
                controller.right(20)
            elif key == 'i':
                controller.up(20)
            elif key == 'k':
                controller.down(20)
            elif key == 'j':
                controller.ccw(20)
            elif key == 'l':
                controller.cw(20)
            elif key == 't':
                controller.takeoff()
            elif key == 'g':
                controller.land()
            elif key == 'h':
                print(help_msg)
            elif key == '\x03':  # Ctrl-C
                controller.land()
                break
            elif key == 'q':  # Quit
                controller.land()
                break
            elif key != '':
                print("Unknown key. Press 'h' for help.")

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()
        restoreTerminalSettings(settings)

