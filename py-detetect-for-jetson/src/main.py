import cv2
import time
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
import psutil
import os
from utils.sort import Sort

# --- CONFIG ---
ENGINE_PATH = 'yolov5s.engine'

# INPUT_SOURCE: 
# - '0': Camera CSI (Ribbon cable)
# - '/dev/video1': Camera USB
# - 'video.mp4': File video
# - 'rtsp://...': Luồng camera IP
INPUT_SOURCE = 'output1.mp4' 

IMG_SIZE = 512
CONF_THRESH = 0.25
IOU_THRESH = 0.45
ROI = [500, 50, 1360, 900] 
LINE_COORDS = [(960, 0), (960, 1080)]

# --- HELPER: GSTREAMER PIPELINE GENERATOR ---
def get_gstreamer_source(source):
    """
    Tạo chuỗi GStreamer pipeline tối ưu cho Jetson Nano dựa trên loại nguồn vào.
    """
    # 1. Camera CSI (Raspberry Pi Cam v2, IMX219) - Dùng NVARGUS
    if source == '0':
        print(">> Phat hien nguon: Camera CSI (NVARGUS)")
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=1280, height=720, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )
    
    # 2. Camera USB (Webcam) - Dùng V4L2
    # Kiểm tra xem có phải là số (video0, video1...) hay đường dẫn thiết bị
    if source.isdigit() or source.startswith('/dev/video'):
        dev_id = source.replace('/dev/video', '') if source.startswith('/dev/video') else source
        print(f">> Phat hien nguon: Camera USB (Device {dev_id})")
        return (
            f"v4l2src device=/dev/video{dev_id} ! "
            "video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )

    # 3. RTSP Stream (Camera IP) - Dùng Hardward Decoder (nvv4l2decoder)
    if source.startswith('rtsp://'):
        print(">> Phat hien nguon: RTSP Stream (Hardware Decoding)")
        return (
            f"rtspsrc location={source} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )

    # 4. File Video (MP4/MKV) - Dùng Hardware Decoder
    if os.path.isfile(source):
        print(f">> Phat hien nguon: File Video '{source}' (Hardware Decoding)")
        return (
            f"filesrc location={source} ! "
            "qtdemux ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
    
    # Fallback: Trả về nguyên gốc nếu không khớp rule nào (để OpenCV tự xử lý)
    print(">> Canh bao: Khong xac dinh ro nguon, dung Fallback.")
    return source

# --- HELPER: GET GPU LOAD ---
def get_jetson_gpu_load():
    try:
        with open("/sys/devices/gpu.0/load", "r") as f:
            load = int(f.read().strip())
            return load / 10.0
    except Exception:
        return 0.0

# --- TENSORRT CLASS ---
class YoLov5TRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            print(f"LOI: Khong tim thay file '{engine_path}'.")
            sys.exit()
            
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = [] 
        self.stream = cuda.Stream()

        if hasattr(self.engine, "num_io_tensors"):
            nb_tensors = self.engine.num_io_tensors
            use_modern_api = True
        else:
            nb_tensors = self.engine.num_bindings
            use_modern_api = False

        for i in range(nb_tensors):
            if use_modern_api:
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            else:
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.allocations.append(int(device_mem))

            if use_modern_api:
                self.context.set_tensor_address(name, int(device_mem))

            tensor_info = {'name': name, 'host': host_mem, 'device': device_mem}
            if is_input: self.inputs.append(tensor_info)
            else: self.outputs.append(tensor_info)

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        if hasattr(self.context, "execute_async_v3"):
             self.context.execute_async_v3(stream_handle=self.stream.handle)
        elif hasattr(self.context, "execute_async_v2"):
             self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream.handle)
        else:
             self.context.execute_v2(bindings=self.allocations)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

# --- UTILS ---
def preprocess_image(img, input_shape):
    h, w, _ = img.shape
    scale = min(input_shape/w, input_shape/h)
    nw, nh = int(scale * w), int(scale * h)
    img_resized = cv2.resize(img, (nw, nh)) 
    image = np.full((input_shape, input_shape, 3), 114, dtype=np.uint8)
    image[:nh, :nw, :] = img_resized
    image = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0), scale, (w, h)

def postprocess_and_merge_classes(output, scale, img_dims, conf_thres, iou_thres):
    pred = output[0].reshape(-1, 85) 
    pred = pred[pred[:, 4] > conf_thres]
    if len(pred) == 0: return []

    boxes = pred[:, :4]
    scores = pred[:, 4]
    class_probs = pred[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    
    mask = np.isin(class_ids, [0]) # Chỉ lấy Person
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(boxes) == 0: return []

    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy /= scale
    
    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), conf_thres, iou_thres)
    final_dets = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            final_dets.append([xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], scores[i]])
            
    return np.array(final_dets)

def is_crossing_line(line_start, line_end, point):
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - \
           (line_end[1] - line_start[1]) * (point[0] - line_start[0])

# --- MAIN ---
def main():
    print("--- SYSTEM INIT WITH GSTREAMER ---")
    model = YoLov5TRT(ENGINE_PATH)
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # --- TÍCH HỢP GSTREAMER ---
    gst_pipeline = get_gstreamer_source(INPUT_SOURCE)
    print(f"Pipeline: {gst_pipeline}")
    
    # Quan trọng: Phải thêm cv2.CAP_GSTREAMER
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("LỖI: Không thể mở Pipeline!")
        print("1. Kiểm tra lại đường dẫn file (cần đường dẫn tuyệt đối hoặc đúng tên).")
        print("2. Nếu dùng file MP4, đảm bảo encode H264 (vì code dùng h264parse).")
        print("3. Nếu dùng Camera CSI, kiểm tra cáp nối.")
        # Thử fallback về backend mặc định nếu Gstreamer thất bại (chỉ dùng cho file/usb)
        if os.path.isfile(INPUT_SOURCE) or INPUT_SOURCE.isdigit():
             print(">> Thử mở lại bằng backend mặc định (CPU decode)...")
             cap = cv2.VideoCapture(INPUT_SOURCE)
    
    if not cap.isOpened():
        sys.exit()

    stats_history = {'fps': [], 'cpu': [], 'gpu': [], 'ram': []}
    count_in = count_out = 0
    prev_pos = {}

    print("--- STARTING LOOP ---")
    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret: 
                print("Ket thuc video hoac mat tin hieu.")
                break

            fh, fw, _ = frame.shape
            safe_roi = [
                max(0, ROI[0]), max(0, ROI[1]), 
                min(fw, ROI[2]), min(fh, ROI[3])
            ]
            
            roi_frame = frame[safe_roi[1]:safe_roi[3], safe_roi[0]:safe_roi[2]]
            
            if roi_frame.size == 0:
                continue

            blob, scale, (orig_w, orig_h) = preprocess_image(roi_frame, IMG_SIZE)
            outputs = model.infer(blob)
            detections = postprocess_and_merge_classes(outputs, scale, (orig_w, orig_h), CONF_THRESH, IOU_THRESH)
            
            if len(detections) > 0:
                detections[:, 0] += safe_roi[0]
                detections[:, 1] += safe_roi[1]
                detections[:, 2] += safe_roi[0]
                detections[:, 3] += safe_roi[1]
                trackers = tracker.update(detections)
            else:
                trackers = tracker.update()

            # Vẽ Line
            cv2.line(frame, LINE_COORDS[0], LINE_COORDS[1], (0, 255, 255), 2)
            cv2.rectangle(frame, (safe_roi[0], safe_roi[1]), (safe_roi[2], safe_roi[3]), (255, 0, 0), 2)

            # Counting Logic
            for d in trackers:
                x1, y1, x2, y2, trk_id = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                
                if trk_id in prev_pos:
                    old_cx, old_cy = prev_pos[trk_id]
                    val_now = is_crossing_line(LINE_COORDS[0], LINE_COORDS[1], (cx, cy))
                    val_old = is_crossing_line(LINE_COORDS[0], LINE_COORDS[1], (old_cx, old_cy))
                    
                    if val_old > 0 and val_now <= 0:
                        count_in += 1
                        cv2.line(frame, LINE_COORDS[0], LINE_COORDS[1], (0, 0, 255), 5)
                    elif val_old <= 0 and val_now > 0:
                        count_out += 1
                        cv2.line(frame, LINE_COORDS[0], LINE_COORDS[1], (255, 0, 255), 5)
                
                prev_pos[trk_id] = (cx, cy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(int(trk_id)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            t_end = time.time()
            time_diff = t_end - t_start
            if time_diff > 0:
                fps = 1.0 / time_diff
            else:
                fps = 30.0 # Giá trị mặc định nếu xử lý quá nhanh
            
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            gpu = get_jetson_gpu_load()
            
            stats_history['fps'].append(fps)
            stats_history['cpu'].append(cpu)
            stats_history['gpu'].append(gpu)
            stats_history['ram'].append(ram)

            info_color = (0, 255, 255)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
            cv2.putText(frame, f"IN: {count_in} | OUT: {count_out}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"CPU: {cpu}% | GPU: {gpu}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow("Jetson Optimized", frame)
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt:
        print("Dung chuong trinh...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # (Phần vẽ biểu đồ giữ nguyên như cũ)

if __name__ == "__main__":
    main()