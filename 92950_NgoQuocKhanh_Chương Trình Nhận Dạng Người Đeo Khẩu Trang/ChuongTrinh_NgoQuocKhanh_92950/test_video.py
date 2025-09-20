import os
import csv
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Cấu hình model YOLO
model_path = "best.pt"
model = YOLO(model_path).to('cuda')

# Hàm chuẩn hóa nhãn
def normalize_label(label):
    return label.split('(')[0].strip().lower()

# Hàm chuẩn hóa tên video
def normalize_filename(filename):
    base_name = os.path.splitext(filename)[0]  # Loại bỏ phần mở rộng
    return base_name.split('(')[0].strip().lower()  # Loại bỏ các phần như "(1)"

# Hàm xử lý video và dự đoán nhãn
def process_video(video_path, conf_threshold):
    if not os.path.exists(video_path):
        print(f"Tệp video không tồn tại: {video_path}")
        return 0, 0, []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return 0, 0, []

    results_list = []
    total_frames = 0
    correct_count = 0

    # Lấy nhãn từ tên video
    video_label = normalize_filename(os.path.basename(video_path))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Dự đoán bằng YOLO
        results = model.predict(source=frame, conf=conf_threshold, device='cuda')
        labels = extract_labels(results)

        # Chuẩn hóa nhãn dự đoán
        normalized_labels = [normalize_label(label) for label in labels]
        predicted_labels = ', '.join(normalized_labels)

        # So sánh nhãn dự đoán với nhãn từ tên video
        match = "Dung" if video_label in normalized_labels else "Sai"
        if match == "Dung":
            correct_count += 1

        # Lưu kết quả từng khung hình
        results_list.append([f"Frame {total_frames}", predicted_labels, match])

    cap.release()
    return correct_count, total_frames, results_list

# Hàm trích xuất nhãn từ kết quả dự đoán
def extract_labels(results):
    labels = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Lấy chỉ số lớp
            label = model.names[cls]  # Lấy nhãn từ lớp
            labels.append(label)
    return labels

# Hàm lưu kết quả ra file CSV
def save_results_to_csv(results, output_file_path):
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Predicted Labels", "Match"])
            writer.writerows(results)
    except PermissionError:
        print(f"Lỗi quyền ghi file tại: {output_file_path}")

# Hàm để chọn folder chứa video và xử lý
def select_folder_and_evaluate_videos():
    # Khởi tạo Tkinter
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    folder_selected = filedialog.askdirectory(title="Chọn thư mục chứa video")
    if folder_selected:
        print(f"Thư mục đã chọn: {folder_selected}")
        video_files = [f for f in os.listdir(folder_selected) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            print(f"Không tìm thấy video trong thư mục: {folder_selected}")
            return

        conf_threshold = 0.5  # Ngưỡng cố định
        total_correct_frames = 0
        total_frames_all_videos = 0
        results_summary = []

        # Xử lý từng video trong thư mục
        for video_file in video_files:
            video_path = os.path.join(folder_selected, video_file)
            print(f"\nĐang xử lý video: {video_file}")

            output_file = os.path.join(folder_selected, f"{os.path.splitext(video_file)[0]}_results.csv")
            correct_count, total_frames, results = process_video(video_path, conf_threshold)

            if total_frames > 0:
                accuracy = (correct_count / total_frames) * 100
                print(f"Tỷ lệ đúng cho video {video_file}: {accuracy:.2f}%")
                results_summary.append([video_file, conf_threshold, accuracy])

                # Tính tổng số khung hình và số khung hình đúng
                total_correct_frames += correct_count
                total_frames_all_videos += total_frames

                # Lưu kết quả từng video vào CSV
                save_results_to_csv(results, output_file)

        # Tính tỷ lệ đúng tổng thể
        if total_frames_all_videos > 0:
            overall_accuracy = (total_correct_frames / total_frames_all_videos) * 100
            print(f"\nTỷ lệ đúng tổng thể của tất cả video: {overall_accuracy:.2f}%")

        # Ghi tóm tắt kết quả vào file CSV
        summary_file = os.path.join(folder_selected, "summary_results.csv")
        with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Video File", "Confidence Threshold", "Accuracy (%)"])
            writer.writerows(results_summary)

        print(f"\nTóm tắt kết quả đã được lưu vào {summary_file}")
    else:
        print("Không có thư mục nào được chọn.")

# Gọi hàm chọn folder và xử lý
if __name__ == "__main__":
    select_folder_and_evaluate_videos()
