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
        
        # 加载基本参数
        model_path = rospy.get_param('~model_path', '/root/model/basketballn_openvino_model_fp32')
        self.conf_thres = rospy.get_param('~conf_threshold', 0.5)
        self.iou_thres = rospy.get_param('~iou_threshold', 0.45)
        self.img_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw/compressed')
        self.queue_size = rospy.get_param('~queue_size', 2) 
        self.num_threads = rospy.get_param('~num_threads', 2)
        
        # 初始化YOLO模型
        self.model = YOLO(model_path, task='detect')
        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres
        self.model.overrides['device'] = 'cpu'
        self.model.overrides['workers'] = 1 

        # 初始化相机参数
        self.enable_undistort = rospy.get_param('~enable_undistort', False)
        self.camera_matrix = None
        self.dist_coeffs = None
        if self.enable_undistort:
            try:
                camera_matrix = rospy.get_param('~camera_matrix')
                self.camera_matrix = np.array(camera_matrix, dtype=np.float32).reshape(3,3)
                dist_coeffs = rospy.get_param('~distortion_coefficients')
                self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).flatten()
                rospy.loginfo("成功加载相机畸变参数")
            except Exception as e:
                rospy.logerr(f"加载相机参数失败: {str(e)}")
                rospy.signal_shutdown("相机参数加载失败")

        # 初始化图像处理
        self.bridge = CvBridge()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.avg_fps = 0.0
        
        self.input_queue = queue.Queue(maxsize=self.queue_size)
        self.output_queue = queue.Queue(maxsize=self.queue_size)

        # 创建处理线程
        self.workers = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            self.workers.append(t)
            
        # 创建发布线程
        self.publisher_thread = threading.Thread(target=self.publish_results)
        self.publisher_thread.daemon = True
        
        # 初始化ROS订阅和发布
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
                
                # 图像解码和预处理
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_image is None:
                    continue
                
                # 畸变矫正
                if self.enable_undistort and self.camera_matrix is not None and self.dist_coeffs is not None:
                    cv_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
                
                # 调整图像尺寸
                resized = cv2.resize(cv_image, (640, 640)) if cv_image.shape[1] > 640 else cv_image
                
                # 执行推理
                results = self.model.predict(resized, imgsz=640, verbose=False)
                # 还原坐标到原始图像尺寸
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
                
                # 筛选每个类别的最高置信度框（添加二次过滤）
                max_conf_boxes = {}
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        # 关键修改：显式过滤低置信度结果
                        if conf < self.conf_thres:
                            continue
                        
                        cls_id = int(box.cls[0])
                        if cls_id not in max_conf_boxes or conf > max_conf_boxes[cls_id][0]:
                            max_conf_boxes[cls_id] = (conf, box)

                # 处理筛选后的检测框
                for cls_id, (conf, box) in max_conf_boxes.items():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = self.model.names[cls_id]
                    
                    # 创建边界框消息
                    bbox = BoundingBox2D()
                    bbox.center.x = (x1 + x2) / 2.0
                    bbox.center.y = (y1 + y2) / 2.0
                    bbox.size_x = x2 - x1
                    bbox.size_y = y2 - y1
                    bbox_array.boxes.append(bbox)
                    
                    # 绘制检测结果
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(debug_image, f"{label}:{conf:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0,255,0), 1)

                # 计算并显示FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    self.avg_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()
                    self.fps_pub.publish(Float32(self.avg_fps))
                
                cv2.putText(debug_image, f"FPS: {self.avg_fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # 发布消息
                self.bbox_pub.publish(bbox_array)
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
                
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