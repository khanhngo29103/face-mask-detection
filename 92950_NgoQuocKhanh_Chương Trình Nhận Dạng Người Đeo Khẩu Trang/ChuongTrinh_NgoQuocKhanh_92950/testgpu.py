import torch

# Kiểm tra tên GPU
print("GPU Name:", torch.cuda.get_device_name(0))

# Kiểm tra GPU đang hoạt động
print("Current Device:", torch.cuda.current_device())

# Kiểm tra khả năng sử dụng GPU
x = torch.rand(3, 3).cuda()  # Tạo một tensor trên GPU
print("Tensor on GPU:", x)
