#  Face Mask Detection (Nhận dạng người đeo khẩu trang)

##  Giới thiệu
Trong bối cảnh dịch Covid-19 và các bệnh hô hấp, việc tuân thủ đeo khẩu trang đóng vai trò quan trọng trong bảo vệ sức khỏe cộng đồng.  
Dự án này xây dựng một **chương trình phát hiện trạng thái đeo khẩu trang** bằng cách áp dụng **thị giác máy tính** và **mô hình YOLOv8**.

Ứng dụng có thể:
- Phát hiện đeo khẩu trang đúng cách  
-  Phát hiện không đeo khẩu trang  
-  Phát hiện đeo khẩu trang sai cách  

Người dùng có thể thử nghiệm qua **ảnh**, **video** hoặc **camera trực tiếp**.

---

##  Công nghệ sử dụng
- Python 3.10  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – huấn luyện và nhận diện  
- OpenCV – xử lý ảnh & video  
- Tkinter – xây dựng giao diện GUI  
- Pillow (PIL) – hiển thị hình ảnh trong giao diện  
- Datetime – hiển thị thời gian thực  

---

##  Cấu trúc chương trình
Chương trình hỗ trợ **3 chế độ nhận dạng**:
1. **Nhận dạng bằng hình ảnh** – tải ảnh và phân tích.  
2. **Nhận dạng bằng video** – chọn video có sẵn để kiểm tra.  
3. **Nhận dạng bằng camera** – sử dụng webcam để phát hiện trực tiếp.  

 Kết quả hiển thị ngay trên giao diện với **bounding box** và **nhãn phân loại**.

---

## 📊 Kết quả mô hình
Mô hình **YOLOv8m** được huấn luyện trên bộ dữ liệu gán nhãn từ **Roboflow**.  
- Precision: **80.38%**  
- Recall: **72.34%**  
- F1-Score: **75.68%**  
- mAP@50: **78.93%**  
- mAP@50-95: **61.43%**

Trên thực nghiệm:
- 🎯 Độ chính xác ảnh: ~**87.32%**  
- 🎯 Độ chính xác video: ~**89.78%**

---
#  Face Mask Detection (Nhận dạng người đeo khẩu trang)

##  Giới thiệu
Trong bối cảnh dịch Covid-19 và các bệnh hô hấp, việc tuân thủ đeo khẩu trang đóng vai trò quan trọng trong bảo vệ sức khỏe cộng đồng.  
Dự án này xây dựng một **chương trình phát hiện trạng thái đeo khẩu trang** bằng cách áp dụng **thị giác máy tính** và **mô hình YOLOv8**.

Ứng dụng có thể:
- Phát hiện đeo khẩu trang đúng cách  
-  Phát hiện không đeo khẩu trang  
-  Phát hiện đeo khẩu trang sai cách  

Người dùng có thể thử nghiệm qua **ảnh**, **video** hoặc **camera trực tiếp**.

---

##  Công nghệ sử dụng
- Python 3.10  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – huấn luyện và nhận diện  
- OpenCV – xử lý ảnh & video  
- Tkinter – xây dựng giao diện GUI  
- Pillow (PIL) – hiển thị hình ảnh trong giao diện  
- Datetime – hiển thị thời gian thực  

---

##  Cấu trúc chương trình
Chương trình hỗ trợ **3 chế độ nhận dạng**:
1. **Nhận dạng bằng hình ảnh** – tải ảnh và phân tích.  
2. **Nhận dạng bằng video** – chọn video có sẵn để kiểm tra.  
3. **Nhận dạng bằng camera** – sử dụng webcam để phát hiện trực tiếp.  

 Kết quả hiển thị ngay trên giao diện với **bounding box** và **nhãn phân loại**.

---

## 📊 Kết quả mô hình
Mô hình **YOLOv8m** được huấn luyện trên bộ dữ liệu gán nhãn từ **Roboflow**.  
- Precision: **80.38%**  
- Recall: **72.34%**  
- F1-Score: **75.68%**  
- mAP@50: **78.93%**  
- mAP@50-95: **61.43%**

Trên thực nghiệm:
- 🎯 Độ chính xác ảnh: ~**87.32%**  
- 🎯 Độ chính xác video: ~**89.78%**

---
## 📦 Dataset
Dữ liệu được tạo và gán nhãn bằng [Roboflow](https://universe.roboflow.com/object-mfpha/facemask-detection-nyuzn/dataset/4).

