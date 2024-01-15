import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np
import cv2
from cv_bridge import CvBridge

class LaneDetect(Node):
    def __init__(self):
        super().__init__('lane_detect')

        self.bridge = CvBridge()
        self.sub_front_image = self.create_subscription(Image, '/car/sensor/camera/front', self.image, 10)

        self.pub_lane_slopes = self.create_publisher(Float32MultiArray, '/lane_slopes', 10)
        self.pub_steering_angle = self.create_publisher(Float32, '/steering_angle', 10)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        self.kernel_size = 3
        self.low_threshold = 90
        self.high_threshold = 270

    def image(self, data):
        self.orgin_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.temp = np.zeros_like(self.orgin_image)

        filter_image = self.filter_colors(self.orgin_image)
        canny_image = self.canny_edge(filter_image)
        roi_image = self.region_of_interest(canny_image)

        self.roi_height, self.roi_width = roi_image.shape[:2]

        hough_image, linesP = self.houghline_tf(roi_image)

        get_L_lines, get_R_lines = self.get_slope_degree(linesP)

        if get_L_lines is not None or get_R_lines is not None:
            left_fit_line = self.get_fitline(self.orgin_image, get_L_lines)
            right_fit_line = self.get_fitline(self.orgin_image, get_R_lines)

            self.draw_fit_line(self.temp, left_fit_line)
            self.draw_fit_line(self.temp, right_fit_line)

            result = self.weighted_img(self.temp, self.orgin_image)

            if left_fit_line is not None or right_fit_line is not None:
                left_angle, right_angle = self.calculate_angle(left_fit_line), self.calculate_angle(right_fit_line)

                avg_slope = (left_angle + right_angle) / 2.0

                steering_angle = avg_slope * 0.5

                lane_angles_msg = Float32MultiArray(data=[left_angle, right_angle])
                self.pub_lane_slopes.publish(lane_angles_msg)
                steering_angle_msg = Float32(data=steering_angle)
                self.pub_steering_angle.publish(steering_angle_msg)

            # cv2.imshow('Lane Detection', result)
            # cv2.waitKey(1)

        else:
            # cv2.imshow('Lane Detection', self.orgin_image)
            # cv2.waitKey(1)
            None

    def filter_colors(self, input_image):
        # BGR을 HSV로 변환
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        # 노란색 및 흰색 범위 정의
        yellow_mask = cv2.inRange(hsv_image, np.array([10, 100, 100]), np.array([40, 255, 255]))
        white_mask1 = cv2.inRange(hsv_image, np.array([110, 20, 150]), np.array([130, 30, 255]))
        white_mask2 = cv2.inRange(hsv_image, np.array([65, 65, 135]), np.array([140, 140, 255]))
        white_mask3 = cv2.inRange(hsv_image, np.array([100, 10, 150]), np.array([150, 30, 200]))

        # HSV 이미지에서 노란색 및 흰색만 얻기
        yellow_dst = cv2.bitwise_and(input_image, input_image, mask=yellow_mask)
        white_dst1 = cv2.bitwise_and(input_image, input_image, mask=white_mask1)
        white_dst2 = cv2.bitwise_and(input_image, input_image, mask=white_mask2)
        white_dst3 = cv2.bitwise_and(input_image, input_image, mask=white_mask3)

        # 흰색 이미지를 가중치를 적용하여 결합
        white_dst_12 = cv2.addWeighted(white_dst1, 1.0, white_dst2, 1.0, 0)
        white_dst_123 = cv2.addWeighted(white_dst_12, 1.0, white_dst3, 1.0, 0)

        # 흰색 & 노란색 이미지를 가중치를 적용하여 결합
        filter_image = cv2.addWeighted(yellow_dst, 1.0, white_dst_123, 1.0, 0)

        return filter_image

    def canny_edge(self, input_image):
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        gaussian_image = cv2.GaussianBlur(gray_image, (self.kernel_size, self.kernel_size), sigmaX=0)
        canny_image = cv2.Canny(gaussian_image, self.low_threshold, self.high_threshold)
        
        return canny_image

    def region_of_interest(self, input_image):
        height, width = input_image.shape[:2]
        zero_mask = np.zeros_like(input_image)

        vertices = np.array([(width/2-70, height/2+30), (width/2+60, height/2+30),
                             (width/2+170, height/2+130), (width/2-180, height/2+130)], dtype=np.int32)
         
        roi_mask = cv2.fillPoly(zero_mask, [vertices], (255, 255, 255))
        roi_image = cv2.bitwise_and(input_image, roi_mask)

        return roi_image

    def houghline_tf(self, input_image):
        cdstP = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        linesP = cv2.HoughLinesP(input_image, 1, np.pi / 180, 40, None, 80, 100)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
            return cdstP, linesP
        
        else:
            # self.get_logger().info("선이 감지되지 않았습니다.")
            return cdstP, None

    def get_slope_degree(self, line_arr):
        if line_arr is not None and len(line_arr) > 0:
            line_arr = np.squeeze(line_arr)

            if len(line_arr.shape) == 2:
                slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

                line_arr = line_arr[np.abs(slope_degree) < 160]
                slope_degree = slope_degree[np.abs(slope_degree) < 160]

                line_arr = line_arr[np.abs(slope_degree) > 95]
                slope_degree = slope_degree[np.abs(slope_degree) > 95]

                L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
                L_lines, R_lines = L_lines[:, None], R_lines[:, None]

            elif len(line_arr.shape) == 1:
                L_lines, R_lines = None, None

            else:
                L_lines, R_lines = None, None

            return L_lines, R_lines
        
        else:
            return None, None

    def get_fitline(self, img, f_lines):
        lines = np.squeeze(f_lines)

        if lines is not None and len(lines.shape) > 0 and lines.shape[0] >= 2:
            lines_flat = lines.reshape(-1, 2)
            
            try:
                output = cv2.fitLine(lines_flat, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x, y = output[0], output[1], output[2], output[3]
                x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x), img.shape[0]-1
                x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x), int(img.shape[0]/2+100)
                result = [x1, y1, x2, y2]

            except cv2.error as e:
                # self.get_logger().error(f"cv2.error: {e}")
                result = None
        else:
            result = None

        return result

    def draw_fit_line(self, img, lines, color=[255, 0, 0], thickness=10):
        if lines is not None:
            cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)
            
        else:
            # self.get_logger().info("선이 감지되지 않았습니다.")
            pass

    def weighted_img(self, img, initial_img, α=1, β=1, λ=0):
        return cv2.addWeighted(initial_img, α, img, β, λ)
    
    def calculate_angle(self, line):
        if line is not None:
            x1, y1, x2, y2 = line
            if (x2 - x1) != 0:
                slope = (y2 - y1) / (x2 - x1)
                angle = np.degrees(np.arctan(slope))
                return angle
            else:
                return 0
        else:
            return 0

def main(args=None):
    rclpy.init(args=args)

    try:
        ldt = LaneDetect()
        rclpy.spin(ldt)

    except KeyboardInterrupt:
        pass

    finally:
        ldt.get_logger().info("차선 검출 노드가 종료되었습니다.")
        ldt.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()