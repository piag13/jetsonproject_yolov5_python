# Jetson Nano People Counting with YOLOv5 & TensorRT

Dự án đếm người và theo dõi chuyển động (Tracking) thời gian thực trên Nvidia Jetson Nano sử dụng YOLOv5 được tăng tốc bởi TensorRT và thuật toán SORT.

## 📋 Yêu cầu phần cứng & Hệ điều hành
* **Thiết bị:** Nvidia Jetson Nano (4GB Developer Kit).
* **Hệ điều hành:** Nvidia JetPack 4.6.1 (Ubuntu 18.04).
* **Camera:** Camera CSI (IMX219) hoặc Webcam USB, hoặc File Video.

## Chuẩn bị 

### Tạo 4GB Swap (tránh OOM khi build / chạy)
Jetson Nano có RAM hạn chế; nếu không tạo Swap, quá trình cài đặt hoặc khi tải model có thể bị treo.

```bash
# 1. Tắt swap hiện tại (nếu có)
sudo swapoff -a

# 2. Tạo file swap 4GB
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile

# 3. Kích hoạt swap
sudo mkswap /swapfile
sudo swapon /swapfile

# 4. Lưu cấu hình để tự động bật sau khi khởi động lại
sudo cp /etc/fstab /etc/fstab.bak
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 5. Kiểm tra
free -h
```
### Cài đặt PyTorch & Torchvision (QUAN TRỌNG)
⚠️ LƯU Ý: KHÔNG dùng lệnh ```pip install torch```. Bạn phải cài bản hỗ trợ GPU (aarch64) từ NVIDIA.

1. Cài đặt PyTorch v1.10.0 (Cho JetPack 4.6):

```Bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```
2. Cài đặt Torchvision v0.11.1:

```Bash
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```

## ⚙️ Hướng dẫn cài đặt môi trường 

### 1. Cập nhật hệ thống và cài đặt các gói cơ bản
Mở Terminal trên Jetson Nano và chạy:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip cmake libopenblas-dev liblapack-dev libjpeg-dev
sudo apt-get install libopencv-dev #nếu chưa có
```
Kiểm tra tensorrt
```bash
dpkg -l | grep tensorrt
```

### 2. Cấu hình biến môi trường CUDA 
Để cài đặt được ```pycuda```, hệ thống cần tìm thấy trình biên địch ```nvcc```.
1. Mở file ```.bashrc```:
```bash
nano ~/.bashrc
```
2. Thêm 2 dòng sau xuống cuối file:
```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
3. Lưu lại (Ctrl+O, Enter) và thoát (Ctrl+X).
4. Cập nhật thay đổi
```bash
source ~/.bashrc
```
5. Kiểm tra (Nếu hiện phiên bản CUDA là thành công):
```bash
nvcc --version
```
### 3. Cài đặt các thư viện Python quan trọng
Lưu ý: **KHÔNG** cài `opencv-python` qua `pip` (JetPack đã có OpenCV phù hợp với GStreamer).
Cũng **KHÔNG** cài `tensorrt` qua `pip` (TensorRT được cung cấp bởi JetPack).

```bash
# Cài PyCUDA (Mất khoảng 10-15 phút để build)
pip3 install pycuda --verbose

# Cài Cython trước (cần cho lap)
pip3 install Cython

# Cài đặt các thư viện toán học và xử lý ảnh
pip3 install numpy>=1.19.4 matplotlib psutil filterpy scipy tqdm pillow jetson-stats

# Cài lap (Linear Assignment Problem) cho thuật toán SORT
pip3 install lap
```

> Tip: nếu gặp lỗi build cho `lap` (linear assignment), cài `Cython` trước và đảm bảo có `python3-dev` / build-essential trên hệ thống.

### 4. Chuẩn bị Model TensorRT
Bạn không thể dùng `.pt` trực tiếp và không nên copy `.engine` từ hệ máy khác. Build engine phải thực hiện trên chính Jetson Nano.

1. Clone YOLOv5 version 6.1: Bắt buộc dùng bản này để tương thích tốt nhất với Python 3.6 trên Nano.

```Bash
# Clone đúng phiên bản v6.1
git clone --branch v6.1 https://github.com/ultralytics/yolov5

# Chỉnh sửa requirements của YOLOv5 để tránh xung đột với PyTorch đã cài
cd yolov5
sed -i 's/torch>=.*/# torch/g' requirements.txt
sed -i 's/torchvision>=.*/# torchvision/g' requirements.txt

# Cài đặt thư viện phụ cho YOLOv5
pip3 install -r requirements.txt
cd ..
```
2. Cài đặt Onnx & Protobuf
```bash
# Protobuf bản 3.20.x là bản ổn định nhất cho Jetson Nano Python 3.6
pip3 install protobuf==3.20.3
pip3 install onnx>=1.9.0
```
#### ⚡ Tối ưu hóa Model (TensorRT)
Để đạt FPS cao, BẮT BUỘC phải chuyển đổi model .pt sang .engine ngay trên Jetson Nano.

1. Tải Model Weights (Phiên bản v6.1):

```Bash
cd yolov5

# Tải YOLOv5s (Small) - Khuyên dùng (Chính xác & Nhanh vừa phải)
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt

# Hoặc Tải YOLOv5n (Nano) - Nếu cần tốc độ cực nhanh (>30 FPS)
# wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```
2. Convert sang Engine (Mất khoảng 15 phút): Chạy lệnh export ngay trên Nano:

```Bash
# Dùng yolov5s (Small) - img size 512
python3 export.py --weights yolov5s.pt --include engine --img 512 --device 0 --half

# Hoặc dùng yolov5n (Nano) - img size 416
# python3 export.py --weights yolov5n.pt --include engine --img 416 --device 0 --half
```

3. Di chuyển file Engine ra thư mục dự án
```bash
mv yolov5s.engine ../models
cd ..
```
### 5. Chạy chương trình
Mở `main.py` và cấu hình `INPUT_SOURCE`:

- Camera CSI (ribbon): `INPUT_SOURCE = '0'`
- Webcam USB: `INPUT_SOURCE = '/dev/video1'` (hoặc device tương ứng)
- File video: `INPUT_SOURCE = 'video.mp4'`

Kích hoạt chế độ hiệu năng cao:
```bash
sudo jetson_clocks
```

Chạy:

```bash
python3 main.py
```

## ▶️ Khắc phục lỗi thường gặp
- ImportError: No module named cv2 — Nguyên nhân: dùng virtualenv/venv thiếu hệ gói, hoặc đã pip cài `opencv-python` đè bản hệ thống. Giải pháp: dùng Python hệ thống (không dùng venv) hoặc `pip3 uninstall opencv-python`.
- SystemError liên quan pycuda — kiểm tra lại `nvcc` và cài `pycuda`.
- OSError: [Errno 12] Cannot allocate memory — thêm Swap (xem mục 0).

## 📂 Cấu trúc thư mục (gợi ý)
```
py-detect-for-jetson/
├── src/
|   └── main.py               # thư mục chứa Mã chạy chính 
├── utils/
│   └── __init__.py
|   └── sort.py            # SORT tracker
├── requirements.txt       # danh sách package để cài trên Jetson
├── models/
|    └── yolov5s.engine   # Engine sinh trên Jetson 
└── README.md
```

---
