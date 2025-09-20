import os
import csv
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Cấu hình model YOLO
model_path = "small.pt"
model = YOLO(model_path).to('cuda')

# Danh sách lớp mục tiêu
classes = ['deo khau trang', 'deo khau trang sai cach', 'khau trang', 'ko deo khau trang', 'undefined']

# Hàm chuẩn hóa nhãn
def normalize_label(label):
    # Loại bỏ các ký tự thừa như "(1)", "(2)",...
    return label.split('(')[0].strip()

# Hàm chuẩn hóa tên file
def normalize_filename(filename):
    base_name = os.path.splitext(filename)[0]  # Loại bỏ phần mở rộng
    return base_name.split('(')[0].strip()  # Loại bỏ phần như (1), (2)

# Hàm xử lý ảnh trong folder và dự đoán nhãn
def process_images_from_folder(folder_path, output_file_path, conf_threshold):
    if not os.path.exists(folder_path):
        print(f"Thư mục không tồn tại: {folder_path}")
        return 0, 0

    # Lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Không tìm thấy ảnh trong thư mục: {folder_path}")
        return 0, 0

    results_list = []
    total_images = len(image_files)
    correct_count = 0

    # Duyệt qua từng ảnh
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            results_list.append([image_file, "Invalid image", "Sai"])
            continue

        # Dự đoán bằng YOLO
        results = model.predict(source=image, conf=conf_threshold, device='cuda')
        labels = extract_labels(results)

        # Chuẩn hóa nhãn
        normalized_labels = [normalize_label(label) for label in labels]
        predicted_labels = ', '.join(normalized_labels)

        # Chuẩn hóa tên file và so sánh
        normalized_filename = normalize_filename(image_file)
        match = "Dung" if normalized_filename in normalized_labels else "Sai"
        if match == "Dung":
            correct_count += 1

        # Lưu kết quả
        results_list.append([image_file, predicted_labels, match])

    # Ghi kết quả vào file CSV
    save_results_to_csv(results_list, output_file_path)
    return correct_count, total_images

# Hàm trích xuất nhãn từ kết quả dự đoán
def extract_labels(results):
    labels = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0]
            label = model.names[cls]
            labels.append(label)
    return labels

# Hàm lưu kết quả ra file CSV
def save_results_to_csv(results, output_file_path):
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Predicted Labels", "Match"])
            writer.writerows(results)
    except PermissionError:
        print(f"Lỗi quyền ghi file tại: {output_file_path}")

# Hàm để chọn folder và chạy với nhiều ngưỡng khác nhau
def select_folder_and_evaluate():
    # Khởi tạo Tkinter
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    folder_selected = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")  # Hiển thị hộp thoại chọn folder
    if folder_selected:
        print(f"Thư mục đã chọn: {folder_selected}")
        best_accuracy = 0
        best_threshold = 0
        results_summary = []

        # Chạy qua các ngưỡng từ 0.0 đến 1.0
        for conf_threshold in [i / 10 for i in range(11)]:
            print(f"Đang xử lý với ngưỡng: {conf_threshold:.1f}")
            output_file = os.path.join(folder_selected, f"results_{conf_threshold:.1f}.csv")
            correct_count, total_images = process_images_from_folder(folder_selected, output_file, conf_threshold)
            if total_images > 0:
                accuracy = (correct_count / total_images) * 100
                print(f"Ngưỡng {conf_threshold:.1f}: Tỷ lệ đúng = {accuracy:.2f}%")
                results_summary.append((conf_threshold, accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = conf_threshold

        # Hiển thị kết quả tốt nhất
        print(f"Ngưỡng tốt nhất: {best_threshold:.1f} với tỷ lệ đúng {best_accuracy:.2f}%")

        # Ghi tóm tắt kết quả
        summary_file = os.path.join(folder_selected, "summary_results.csv")
        with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Confidence Threshold", "Accuracy (%)"])
            writer.writerows(results_summary)

        print(f"Tóm tắt kết quả đã được lưu vào {summary_file}")
    else:
        print("Không có thư mục nào được chọn.")

# Gọi hàm chọn folder và xử lý
if __name__ == "__main__":
    select_folder_and_evaluate()