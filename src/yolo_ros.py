#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
import threading
import queue
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray
from std_msgs.msg import Float32

class YOLOv11DetectorCPU:
    def __init__(self):
        rospy.init_node('yolo11_detector_cpu', anonymous=True)
        
        model_path = rospy.get_param('~model_path', '/root/')
        self.conf_thres = rospy.get_param('~conf_threshold', 0.5)
        self.iou_thres = rospy.get_param('~iou_threshold', 0.45)
        self.img_topic = rospy.get_param('~image_topic', '/usb_cam/image/compressed')
        self.queue_size = rospy.get_param('~queue_size', 2) 
        self.num_threads = rospy.get_param('~num_threads', 2)
        
        self.model = YOLO(model_path, task='detect')
        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres
        self.model.overrides['device'] = 'cpu'
        self.model.overrides['workers'] = 1 

        self.bridge = CvBridge()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.avg_fps = 0.0
        
        self.input_queue = queue.Queue(maxsize=self.queue_size)
        self.output_queue = queue.Queue(maxsize=self.queue_size)
        
        self.workers = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            self.workers.append(t)
            
        self.publisher_thread = threading.Thread(target=self.publish_results)
        self.publisher_thread.daemon = True
        
        self.image_sub = rospy.Subscriber(
            self.img_topic, 
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=512 * 1024
        )
        
        self.bbox_pub = rospy.Publisher('/yolo/detections', BoundingBox2DArray, queue_size=5)
        self.debug_pub = rospy.Publisher('/yolo/debug_image', Image, queue_size=2)
        self.fps_pub = rospy.Publisher('/yolo/fps', Float32, queue_size=1)
        
        self.publisher_thread.start()
        
        rospy.on_shutdown(self.shutdown_handler)

    def shutdown_handler(self):
        self.image_sub.unregister()
        for t in self.workers:
            if t.is_alive():
                t.join(timeout=1)
        self.publisher_thread.join(timeout=1)
        rospy.loginfo("节点已安全关闭")

    def image_callback(self, msg):
        try:
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
            self.input_queue.put_nowait(msg)
        except Exception as e:
            rospy.logwarn(f"入队失败: {str(e)}")

    def worker(self):
        while not rospy.is_shutdown():
            try:
                msg = self.input_queue.get(timeout=0.5)
                np_arr = np.frombuffer(msg.data, np.uint8)
                
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_image is None:
                    continue
                
                resized = cv2.resize(cv_image, (640, 640)) if cv_image.shape[1] > 640 else cv_image
                
                results = self.model.predict(resized, imgsz=640, verbose=False)
                
                if resized.shape != cv_image.shape:
                    for r in results:
                        r.boxes.xyxy *= np.array([cv_image.shape[1]/resized.shape[1], 
                                                cv_image.shape[0]/resized.shape[0]]*2)

                self.output_queue.put((msg.header, cv_image, results), timeout=0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"处理异常: {str(e)}")

    def publish_results(self):
        fps_counter = 0
        fps_start = time.time()
        
        while not rospy.is_shutdown():
            try:
                header, cv_image, results = self.output_queue.get(timeout=0.5)
                debug_image = cv_image.copy()
                
                bbox_array = BoundingBox2DArray()
                bbox_array.header = header
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        
                        bbox = BoundingBox2D()
                        bbox.center.x = (x1 + x2) / 2.0
                        bbox.center.y = (y1 + y2) / 2.0
                        bbox.size_x = x2 - x1
                        bbox.size_y = y2 - y1
                        bbox_array.boxes.append(bbox)
                        
                        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(debug_image, f"{label}:{conf:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0,255,0), 1)
                
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    self.avg_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()
                
                cv2.putText(debug_image, f"FPS: {self.avg_fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                self.bbox_pub.publish(bbox_array)
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
                self.fps_pub.publish(Float32(self.avg_fps))
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"发布异常: {str(e)}")

if __name__ == '__main__':
    try:
        detector = YOLOv11DetectorCPU()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass