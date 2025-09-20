import os
import csv
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Cấu hình model YOLO
model_path = "best.pt"
model = YOLO(model_path).to('cuda')

# Danh sách lớp mục tiêu (Ground Truth Labels)
ground_truth = {
    # Cấu trúc: ("Tên video", Số khung hình): [Danh sách nhãn thực]
    ("video1.mp4", 1): ["deo khau trang"],
    ("video1.mp4", 2): ["ko deo khau trang"],
    ("video2.avi", 1): ["deo khau trang sai cach"],
}

# Ngưỡng độ tin cậy cố định
confidence_threshold = 0.5


# Hàm trích xuất nhãn từ kết quả dự đoán
def extract_labels(results):
    labels = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                label = model.names[cls]
                labels.append(label)
    return labels


# Hàm kiểm tra độ chính xác của khung hình
def check_frame_accuracy(predicted_labels, ground_truth_labels):
    if not ground_truth_labels:
        return False
    return all(label in ground_truth_labels for label in predicted_labels)


# Hàm tính toán độ chính xác tổng thể
def calculate_accuracy(output_results, ground_truth):
    correct = 0
    total = 0

    for result in output_results[1:]:  # Bỏ qua tiêu đề
        video_name, frame_id, predicted_labels = result
        predicted_labels = predicted_labels.split(", ")

        # Lấy nhãn thực từ Ground Truth
        frame_id = int(frame_id)
        true_labels = ground_truth.get((video_name, frame_id), [])

        if check_frame_accuracy(predicted_labels, true_labels):
            correct += 1
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy


# Hàm xử lý một video
def process_single_video(video_path, output_results):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    frame_id = 0
    video_name = os.path.basename(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Dự đoán bằng YOLO
        results = model.predict(source=frame, conf=confidence_threshold, device='cuda')
        predicted_labels = extract_labels(results)

        # Lưu kết quả cho khung hình
        output_results.append([video_name, frame_id, ', '.join(predicted_labels)])

    cap.release()


# Hàm xử lý tất cả video trong thư mục
def process_all_videos_in_folder(folder_path, output_csv_path):
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi'))]
    if not video_files:
        print(f"Không có video nào trong thư mục: {folder_path}")
        return

    output_results = [["Video Name", "Frame ID", "Predicted Labels"]]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Đang xử lý video: {video_path}")
        process_single_video(video_path, output_results)

    # Ghi tất cả kết quả vào file CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_results)
        print(f"Kết quả đã được lưu vào: {output_csv_path}")
    except PermissionError:
        print(f"Lỗi quyền ghi file tại: {output_csv_path}")

    # Tính toán độ chính xác
    accuracy = calculate_accuracy(output_results, ground_truth)
    print(f"Độ chính xác trên toàn bộ video: {accuracy:.2f}%")


# Hàm chọn thư mục và xử lý
def select_folder_and_process_videos():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    folder_selected = filedialog.askdirectory(title="Chọn thư mục chứa video")
    if folder_selected:
        print(f"Thư mục đã chọn: {folder_selected}")
        output_csv_path = os.path.join(folder_selected, "all_videos_results.csv")
        process_all_videos_in_folder(folder_selected, output_csv_path)
    else:
        print("Không có thư mục nào được chọn.")


# Gọi hàm chọn thư mục và xử lý
if __name__ == "__main__":
    select_folder_and_process_videos()
