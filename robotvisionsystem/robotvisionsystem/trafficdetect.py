import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficDetect(Node):
    def __init__(self, cascade_classifier_path):
        super().__init__('traffic_detect')

        self.bridge = CvBridge()
        self.sub_front_image = self.create_subscription(Image, '/car/sensor/camera/front', self.image_callback, 10)

        self.signal_publisher = self.create_publisher(Int32, '/traffic_signal', 10)

        self.signal_msg = Int32()

        self.classifier = cv2.CascadeClassifier(cascade_classifier_path)
        self.check_classifier()

    def check_classifier(self):
        if self.classifier.empty():
            self.get_logger().error("카스케이드 분류기를 로드하는 중 오류가 발생했습니다.")
            raise RuntimeError("카스케이드 분류기를 로드할 수 없습니다.")

    def image_callback(self, data):
        origin_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        roi_image = self.region_of_interest(origin_image)
        traffic_detect = self.cascade_classifier(roi_image)

        if traffic_detect is not None:
            # cv2.imshow('Traffic Detection', traffic_detect)
            # cv2.waitKey(1)
            None
            
        else:
            # cv2.imshow('Traffic Detection', origin_image)
            # cv2.waitKey(1)
            None

    def region_of_interest(self, input_image):
        height, width = input_image.shape[:2]
        zero_mask = np.zeros_like(input_image)

        vertices = np.array([(width/2-100, height/3-20), (width/2+100, height/3-20),
                             (width/2+100, height/3+55), (width/2-100, height/3+55)], dtype=np.int32)
         
        roi_mask = cv2.fillPoly(zero_mask, [vertices], (255, 255, 255))
        roi_image = cv2.bitwise_and(input_image, roi_mask)

        return roi_image

    def cascade_classifier(self, input_image):
        if self.classifier.empty():
            return input_image

        traffic_light = self.classifier.detectMultiScale(input_image,minNeighbors=4)
        result = input_image

        for (x, y, w, h) in traffic_light:
            roi = input_image[y:y+h, x:x+w]

            self.light_detect(roi)

            result = cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 255), 2)

        return result

    def light_detect(self, input_image):
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image, np.array([170, 100, 200]), np.array([180, 255, 255]))
        yellow_mask = cv2.inRange(hsv_image, np.array([20, 100, 200]), np.array([40, 255, 255]))
        green_mask = cv2.inRange(hsv_image, np.array([40, 100, 200]), np.array([70, 255, 255]))

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        if green_pixels > yellow_pixels and green_pixels > red_pixels:
            # self.get_logger().info("녹색 신호")
            self.signal_msg.data = 2

        elif yellow_pixels > green_pixels and yellow_pixels > red_pixels:
            # self.get_logger().info("노란색 신호")
            self.signal_msg.data = 1

        elif red_pixels > green_pixels and red_pixels > yellow_pixels:
            # self.get_logger().info("빨간색 신호")
            self.signal_msg.data = 0

        self.signal_publisher.publish(self.signal_msg)

def main(args=None):
    rclpy.init(args=args)
    cascade_classifier_path = 'src/robotvisionsystem/cascade_train/classifier/cascade.xml'

    try:
        tdt = TrafficDetect(cascade_classifier_path)
        rclpy.spin(tdt)

    except KeyboardInterrupt:
        pass

    finally:
        tdt.get_logger().info("신호 검출 노드가 종료되었습니다.")
        tdt.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()