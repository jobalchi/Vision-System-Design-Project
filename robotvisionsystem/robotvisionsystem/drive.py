import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, Int32
from robotvisionsystem_msgs.msg import Motor, State

import threading
import time

class AutonomousDrive(Node):
    def __init__(self):
        super().__init__('lane_keeping_system')

        self.POS_X_POINT1_RANGE = [1040,1055]
        self.POS_X_POINT2_RANGE = [827,836]
        self.POS_X_POINT3_RANGE = [893,910]
        self.POS_X_POINT4_RANGE = [1110,1113]

        self.POS_Y_POINT1_RANGE = [10,12]
        self.POS_Y_POINT2_RANGE = [13,14]
        self.POS_Y_POINT3_RANGE = [14,16]
        self.POS_Y_POINT4_RANGE = [9999,9999]

        self.POS_Z_POINT1_RANGE = [10,12]
        self.POS_Z_POINT2_RANGE = [2440,2456]
        self.POS_Z_POINT3_RANGE = [2290,2306]
        self.POS_Z_POINT4_RANGE = [9999,9999]

        self.sub_steering_angle = self.create_subscription(Float32, '/steering_angle', self.steering_angle_callback, 10)
        self.sub_traffic_signal = self.create_subscription(Int32, '/traffic_signal', self.traffic_signal_callback, 10)
        self.sub_stop_detected = self.create_subscription(Bool, '/stop_detected', self.stop_detected_callback, 10)
        self.sub_car_state = self.create_subscription(State, '/car/state', self.car_state_callback, 10)

        self.pub_motor = self.create_publisher(Motor, '/car/motor', 10)

        self.motor = Motor()
        self.pid_output = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.kp = 0.2
        self.ki = 0.0
        self.kd = 0.12

        self.steering_angle = 0.0
        self.stop_detected = False
        self.traffic_signal = -1

        self.timer = self.create_timer(0.01, self.drive_algorithm)

    def car_state_callback(self, msg):
        self.pos_x = msg.pos_x
        self.pos_y = msg.pos_y
        self.pos_z = msg.pos_z

    def steering_angle_callback(self, msg):
        self.steering_angle = msg.data

    def traffic_signal_callback(self, msg):
        self.traffic_signal = msg.data

    def stop_detected_callback(self, msg):
        self.stop_detected = msg.data

    def calculate_pid_output(self):
        error = 0.0 - self.steering_angle
        self.integral += error
        self.pid_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * (error - self.prev_error))
        self.prev_error = error

    def update_motor_command(self):
        self.motor.steer = self.pid_output
        self.motor.motorspeed = 0.08
        self.motor.breakbool = False
        
        if self.traffic_signal == 0 and self.stop_detected:
            if ((self.POS_X_POINT1_RANGE[0] <= self.pos_x <= self.POS_X_POINT1_RANGE[1] and self.POS_Y_POINT1_RANGE[0] <= self.pos_y <= self.POS_Y_POINT1_RANGE[1]) or
                (self.POS_X_POINT2_RANGE[0] <= self.pos_x <= self.POS_X_POINT2_RANGE[1] and self.POS_Y_POINT2_RANGE[0] <= self.pos_y <= self.POS_Y_POINT2_RANGE[1]) or
                (self.POS_X_POINT3_RANGE[0] <= self.pos_x <= self.POS_X_POINT3_RANGE[1] and self.POS_Y_POINT3_RANGE[0] <= self.pos_y <= self.POS_Y_POINT3_RANGE[1]) or
                (self.POS_X_POINT4_RANGE[0] <= self.pos_x <= self.POS_X_POINT4_RANGE[1] and self.POS_Y_POINT4_RANGE[0] <= self.pos_y <= self.POS_Y_POINT4_RANGE[1])):
                self.motor.motorspeed = 0.0
                self.motor.steer = 0.0
                self.motor.breakbool = True

        elif self.traffic_signal == 1:
            if self.POS_X_POINT1_RANGE[0] <= self.pos_x <= self.POS_X_POINT1_RANGE[1] and self.POS_Y_POINT1_RANGE[0] <= self.pos_y <= self.POS_Y_POINT1_RANGE[1]:
                None

            elif self.POS_X_POINT2_RANGE[0] <= self.pos_x <= self.POS_X_POINT2_RANGE[1] and self.POS_Y_POINT2_RANGE[0] <= self.pos_y <= self.POS_Y_POINT2_RANGE[1]:
                None

            elif self.POS_X_POINT3_RANGE[0] <= self.pos_x <= self.POS_X_POINT3_RANGE[1] and self.POS_Y_POINT3_RANGE[0] <= self.pos_y <= self.POS_Y_POINT3_RANGE[1]:
                self.motor.motorspeed = 0.02
                self.motor.steer = 0.0
                self.motor.breakbool = False
                self.stop_detected = False

        elif self.traffic_signal == 2:
            if self.POS_X_POINT1_RANGE[0] <= self.pos_x <= self.POS_X_POINT1_RANGE[1] and self.POS_Y_POINT1_RANGE[0] <= self.pos_y <= self.POS_Y_POINT1_RANGE[1]:
                self.motor.motorspeed = 0.12
                self.motor.steer = 0.0
                self.motor.breakbool = False
                self.stop_detected = False

            elif self.POS_X_POINT2_RANGE[0] <= self.pos_x <= self.POS_X_POINT2_RANGE[1] and self.POS_Y_POINT2_RANGE[0] <= self.pos_y <= self.POS_Y_POINT2_RANGE[1] and self.POS_Z_POINT2_RANGE[0] <= self.pos_z <= self.POS_Z_POINT2_RANGE[1]:
                self.motor.motorspeed = 0.2
                self.motor.steer = -13.0
                self.motor.breakbool = False
                self.stop_detected = False

            elif self.POS_X_POINT3_RANGE[0] <= self.pos_x <= self.POS_X_POINT3_RANGE[1] and self.POS_Y_POINT3_RANGE[0] <= self.pos_y <= self.POS_Y_POINT3_RANGE[1] and self.POS_Z_POINT3_RANGE[0] <= self.pos_z <= self.POS_Z_POINT3_RANGE[1]:
                self.motor.motorspeed = 0.3
                self.motor.steer = -13.0
                self.motor.breakbool = False
                self.stop_detected = False

            elif self.POS_X_POINT4_RANGE[0] <= self.pos_x <= self.POS_X_POINT4_RANGE[1] and self.POS_Y_POINT4_RANGE[0] <= self.pos_y <= self.POS_Y_POINT4_RANGE[1]:
                None

        self.pub_motor.publish(self.motor)

    def drive_algorithm(self):
        self.calculate_pid_output()
        self.update_motor_command()

def main(args=None):
    rclpy.init(args=args)

    try:
        ad = AutonomousDrive()
        rclpy.spin(ad)

    finally:
        ad.get_logger().info("자율 주행 시스템이 종료되었습니다.")
        ad.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()