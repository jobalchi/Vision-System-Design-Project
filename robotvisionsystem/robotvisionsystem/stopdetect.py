import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import numpy as np
import cv2
from cv_bridge import CvBridge

class StopDetect(Node):
    def __init__(self):
        super().__init__('stop_detect')

        self.bridge = CvBridge()
        self.sub_front_image = self.create_subscription(Image, '/car/sensor/camera/front', self.image, 10)

        self.pub_stop_detected = self.create_publisher(Bool, '/stop_detected', 10)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        self.KERNEL_SIZE = 3
        self.LOW_THRESHOLD = 90
        self.HIGH_THRESHOLD = 270

    def image(self, data):
        self.origin_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        filter_image = self.filter_colors(self.origin_image)
        canny_image = self.canny_edge(filter_image)
        roi_image = self.region_of_interest(canny_image)

        self.roi_height, self.roi_width = roi_image.shape[:2]

        birdseye_image = self.birdseye_view(roi_image)

        birdseye_roi = birdseye_image[self.roi_height//2:, :]

        stop_lane_detect, lines = self.houghline_tf(birdseye_roi)

        if lines is not None:
            stop_detected_msg = Bool()
            stop_detected_msg.data = True
            self.pub_stop_detected.publish(stop_detected_msg)

        # cv2.imshow('Stop Lane Detection', stop_lane_detect)
        # cv2.waitKey(1)

    def filter_colors(self, input_image):
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        gray_mask = cv2.inRange(hsv_image, (100, 0, 100), (179, 20, 200))
        white_mask = cv2.inRange(hsv_image, (110, 20, 150), (130, 30, 255))

        mask = gray_mask + white_mask

        stop_dst = cv2.bitwise_and(input_image, input_image, mask=mask)

        return stop_dst

    def canny_edge(self, input_image):
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        gaussian_image = cv2.GaussianBlur(gray_image, (self.KERNEL_SIZE, self.KERNEL_SIZE), sigmaX=0)
        canny_image = cv2.Canny(gaussian_image, self.LOW_THRESHOLD, self.HIGH_THRESHOLD)
        
        return canny_image

    def region_of_interest(self, input_image):
        height, width = input_image.shape[:2]
        zero_mask = np.zeros_like(input_image)

        vertices = np.array([(width/2-60, height/2+30),
                             (width/2+50, height/2+30),
                             (width/2+160, height/2+130),
                             (width/2-170, height/2+130)], dtype=np.int32)
         
        roi_mask = cv2.fillPoly(zero_mask, [vertices], (255, 255, 255))
        roi_image = cv2.bitwise_and(input_image, roi_mask)

        return roi_image

    def birdseye_view(self, input_image):
        src = np.float32([[self.roi_width/2-60, self.roi_height/2+30], 
                          [self.roi_width/2+50, self.roi_height/2+30], 
                          [self.roi_width/2+160, self.roi_height/2+130], 
                          [self.roi_width/2-170, self.roi_height/2+130]])
        
        dst = np.float32([[0, 0], 
                          [self.roi_width, 0], 
                          [self.roi_width, self.roi_height], 
                          [0, self.roi_height]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(input_image, M, (self.roi_width, self.roi_height))

        return warped_img
    
    def houghline_tf(self, input_image):
        cdstP = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        linesP = cv2.HoughLinesP(input_image, 1, np.pi / 180, 40, None, 500, 100)

        if linesP is not None:
            longest_line = max(linesP, key=lambda line: np.linalg.norm(np.array(line[0][:2]) - np.array(line[0][2:])))
            l = longest_line[0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)

            return cdstP, linesP
        
        else:
            # self.get_logger().info("선이 감지되지 않았습니다.")
            return cdstP, None

def main(args=None):
    rclpy.init(args=args)

    try:
        sdt = StopDetect()
        rclpy.spin(sdt)

    except KeyboardInterrupt:
        pass

    finally:
        sdt.get_logger().info("정지선 검출 노드가 종료되었습니다.")
        sdt.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()