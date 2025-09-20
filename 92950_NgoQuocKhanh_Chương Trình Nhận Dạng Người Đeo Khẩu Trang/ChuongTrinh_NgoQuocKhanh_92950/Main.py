import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import datetime
from ultralytics import YOLO

# Biến trạng thái
running = False
cap = None
image_path = ""
video_path = ""
confidence_threshold = 0.5
trigger_recognition = False
playback_speed = 1

model_path =  "nano.pt"
# model_path = "model/best1.1.pt"
model = YOLO(model_path).to('cuda')

# Hàm hiển thị ảnh hoặc khung hình từ video lên Tkinter
def show_image(image, label):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image.thumbnail((label.winfo_width(), label.winfo_height()))
    tk_image = ImageTk.PhotoImage(pil_image)
    label.config(image=tk_image)
    label.image = tk_image

# Hàm nhận dạng trên ảnh và hiển thị kết quả
def result_image():
    if image_path:
        image = cv2.imread(image_path)
        results = model.predict(source=image, conf=confidence_threshold, device='cuda')  # Chạy trên GPU
        labels = extract_labels(results)
        show_image_with_results(image, results)
        update_status(f"Nhận dạng hoàn tất: {labels}")

# Hàm trích xuất nhãn từ kết quả nhận dạng
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
    return ', '.join(labels)


def draw_results(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == 'deo khau trang':
                    color = (0, 255, 0)
                elif label == 'ko deo khau trang':
                    color = (0, 0, 255)
                elif label == 'deo khau trang sai cach':
                    color = (0, 255, 255)
                else:
                    color = (255, 0, 255)

                if color == (0, 0, 255):
                    text_color = (255, 255, 255)
                elif color in [(0, 255, 0), (0, 255, 255)]:
                    text_color = (0, 0, 0)  # Chữ đen
                else:
                    text_color = (255, 255, 255)

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Chuẩn bị văn bản
                text = f"{label} {confidence:.2f}"

                box_width = x2 - x1
                font_scale = max(0.5, min(box_width / 300, 2))
                font_thickness = max(1, int(font_scale))

                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                      font_thickness)

                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

                cv2.rectangle(image, (text_x, text_y - text_height - baseline),
                              (text_x + text_width, text_y + baseline), color, cv2.FILLED)

                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                            font_thickness, cv2.LINE_AA)

    return image



# Hàm hiển thị ảnh với kết quả nhận dạng
def show_image_with_results(image, results):
    image_with_results = draw_results(image.copy(), results)
    show_image(image_with_results, img_label_right)

# Hàm để mở và phát video
def open_and_play_video():
    global video_path, cap, running
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        running = True
        update_status("Đang phát video")

        def update_frame():
            global running
            if not running or not cap.isOpened():
                cap.release()
                return
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                if trigger_recognition:
                    results = model.predict(source=frame, conf=confidence_threshold, device='cuda')  # Chạy trên GPU
                    frame = draw_results(frame, results)
                show_image(frame, img_label_left)
                img_label_left.after(int(10 / playback_speed), update_frame)
            else:
                cap.release()
                update_status("Phát video hoàn tất")
                running = False

        update_frame()

# Hàm bật/tắt nhận dạng
def toggle_recognition():
    global trigger_recognition
    trigger_recognition = not trigger_recognition
    update_status("Nhận dạng đang bật" if trigger_recognition else "Nhận dạng đã tắt")

# Hàm mở ảnh
def open_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        path_entry.config(state='normal')
        path_entry.delete(0, tk.END)
        path_entry.insert(0, image_path)
        path_entry.config(state='readonly')
        image = cv2.imread(image_path)
        show_image(image, img_label_left)
        update_status("Đã tải ảnh thành công")

# Hàm làm mới ứng dụng
def refresh_app():
    global running, image_path, video_path, trigger_recognition
    stop_camera()
    image_path = ""
    video_path = ""
    trigger_recognition = False
    img_label_left.config(image='')
    img_label_right.config(image='')
    path_entry.config(state='normal')
    path_entry.delete(0, tk.END)
    path_entry.config(state='readonly')
    update_status("Ứng dụng đã được làm mới")
    combo.set("chọn chế độ nhận dạng")  # Đặt lại ComboBox về giá trị ban đầu

# Hàm cập nhật trạng thái
def update_status(message):
    status_label.config(text=f"TRẠNG THÁI: {message}")

# Hàm xác nhận chế độ
def confirm_mode():
    selected_option = combo.get()
    if selected_option == "chọn chế độ nhận dạng":
        update_status("Yêu cầu chọn chế độ nhận dạng")
        return
    refresh_app()
    clear_frames()
    if selected_option == "Nhận dạng bằng hình ảnh":
        picture_controls.pack(side=tk.TOP, pady=10)
        image_frame_left.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)
        image_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill="both", expand=True)
    elif selected_option == "Nhận dạng bằng camera":
        camera_controls.pack(side=tk.TOP, pady=10)
        image_frame_left.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)
    elif selected_option == "Nhận dạng bằng video":
        video_controls.pack(side=tk.TOP, pady=10)
        image_frame_left.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)
# Hàm xóa frame
def clear_frames():
    picture_controls.pack_forget()
    camera_controls.pack_forget()
    video_controls.pack_forget()
    image_frame_left.pack_forget()
    image_frame_right.pack_forget()

# Hàm khởi động camera
def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    update_status("Camera đã khởi động")

    def update_camera_frame():
        if running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if trigger_recognition:
                    results = model.predict(source=frame, conf=confidence_threshold, device='cuda')  # Chạy trên GPU
                    frame = draw_results(frame, results)
                show_image(frame, img_label_left)
                img_label_left.after(10, update_camera_frame)
            else:
                stop_camera()
        else:
            stop_camera()

    update_camera_frame()

# Hàm dừng camera
def stop_camera():
    global cap, running
    if cap and cap.isOpened():
        cap.release()
    running = False
    img_label_left.config(image='')
    update_status("Camera đã dừng")

# Hàm cập nhật ngưỡng độ tin cậy
def update_confidence(value):
    global confidence_threshold
    confidence_threshold = float(value)
    update_status(f"Ngưỡng độ tin cậy: {confidence_threshold}")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Chương trình nhận dạng khẩu trang")
root.state('zoomed')

# Frame trạng thái
status_frame = tk.Frame(root, bg="blue")
status_frame.pack(fill="x")
status_label = tk.Label(status_frame, text="TRẠNG THÁI: OK", font=('Helvetica', 12, 'bold'), bg="blue", fg="white")
status_label.pack(side=tk.LEFT, padx=10)

# Hiển thị thời gian
def update_time():
    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    time_label.config(text=current_time)
    root.after(1000, update_time)

time_label = tk.Label(status_frame, font=('calibri', 12, 'bold'), background='blue', foreground='white')
time_label.pack(side=tk.RIGHT, padx=10)
update_time()

# Frame chính
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Frame cho ảnh/video gốc và kết quả
image_frame_left = tk.Frame(main_frame, bd=2, relief="sunken", width=400, height=600)
image_frame_right = tk.Frame(main_frame, bd=2, relief="sunken", width=400, height=600)
img_label_left = tk.Label(image_frame_left)
img_label_left.pack(fill="both", expand=True)
img_label_right = tk.Label(image_frame_right)
img_label_right.pack(fill="both", expand=True)

# Frame cho các cài đặt và điều khiển
control_frame = tk.Frame(main_frame, width=300)
control_frame.pack(side=tk.RIGHT, fill="y")

# Combobox để chọn chế độ nhận dạng
mode_label = tk.Label(control_frame, text="Mode :", font=('Helvetica', 12, 'bold'))
mode_label.pack(pady=5)
combo_options = ["Nhận dạng bằng hình ảnh", "Nhận dạng bằng camera", "Nhận dạng bằng video"]
combo = ttk.Combobox(control_frame, values=combo_options, width=25, font=('Helvetica', 12))
combo.set("chọn chế độ nhận dạng")
combo.pack()
confirm_button = tk.Button(control_frame, text="XÁC NHẬN", command=confirm_mode, width=20, bg="red", fg="white", font=('Helvetica', 12))
confirm_button.pack(pady=5)

picture_controls = tk.Frame(control_frame)
path_label = tk.Label(picture_controls, text="Đường dẫn :", font=('Helvetica', 12))
path_label.pack()
path_entry = tk.Entry(picture_controls, width=30, state='readonly', font=('Helvetica', 12))
path_entry.pack()
choose_button = tk.Button(picture_controls, text="CHỌN ẢNH", width=20, command=open_image, bg="red", fg="white", font=('Helvetica', 12))
choose_button.pack(pady=5)
recognize_button = tk.Button(picture_controls, text="NHẬN DẠNG", width=20, command=result_image, bg="green", fg="white", font=('Helvetica', 12))
recognize_button.pack(pady=5)

# Frame cho các điều khiển cho nhận dạng từ camera
camera_controls = tk.Frame(control_frame)
camera_label = tk.Label(camera_controls, text="Chọn camera:", font=('Helvetica', 12))
camera_label.pack()
camera_combo = ttk.Combobox(camera_controls, values=["Camera 0"], width=25, font=('Helvetica', 12))
camera_combo.set("Camera 0")
camera_combo.pack()
start_button = tk.Button(camera_controls, text="Bắt đầu camera", width=20, command=start_camera, bg="green", fg="white", font=('Helvetica', 12))
start_button.pack(pady=5)
stop_button = tk.Button(camera_controls, text="Dừng camera", width=20, command=stop_camera, bg="red", fg="white", font=('Helvetica', 12))
stop_button.pack(pady=5)
recognition_button = tk.Button(camera_controls, text="Kích hoạt Nhận dạng", width=20, command=toggle_recognition, bg="yellow", fg="black", font=('Helvetica', 12))
recognition_button.pack(pady=5)

# Frame cho các điều khiển cho nhận dạng từ video
video_controls = tk.Frame(control_frame)
choose_play_video_button = tk.Button(video_controls, text="CHỌN VÀ PHÁT VIDEO", width=20, command=open_and_play_video, bg="green", fg="white", font=('Helvetica', 12))
choose_play_video_button.pack(pady=5)
recognize_video_button = tk.Button(video_controls, text="BẬT/TẮT NHẬN DẠNG", width=20, command=toggle_recognition, bg="yellow", fg="black", font=('Helvetica', 12))
recognize_video_button.pack(pady=5)

# Điều khiển ngưỡng độ tin cậy
confidence_label = tk.Label(control_frame, text="Ngưỡng độ tin cậy:", font=('Helvetica', 12))
confidence_label.pack()
confidence_scale = tk.Scale(control_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=200, variable=tk.DoubleVar(value=confidence_threshold), command=update_confidence)
confidence_scale.pack()

# Nút Làm Mới và Thoát
refresh_button = tk.Button(control_frame, text="LÀM MỚI", command=refresh_app, width=20, bg="blue", fg="white", font=('Helvetica', 12))
refresh_button.pack(pady=10)
exit_button = tk.Button(control_frame, text="THOÁT", command=root.quit, width=20, bg="red", fg="white", font=('Helvetica', 12))
exit_button.pack(pady=10)

root.mainloop()
