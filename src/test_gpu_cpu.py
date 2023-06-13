import torch
import time

# Size of square matrix
size = 1000

# Creating random tensors for CPU
x_cpu = torch.rand(size, size)
y_cpu = torch.rand(size, size)

# Perform operation on CPU
start_time_cpu = time.time()

z_cpu = torch.mm(x_cpu, y_cpu)

end_time_cpu = time.time()

print(f"Time on CPU: {end_time_cpu - start_time_cpu} seconds.")

# Check if a CUDA device is available
if torch.cuda.is_available():
    # Creating random tensors for GPU
    x_gpu = torch.rand(size, size).cuda()
    y_gpu = torch.rand(size, size).cuda()

    # Perform operation on GPU
    start_time_gpu = time.time()

    z_gpu = torch.mm(x_gpu, y_gpu)

    # Wait for GPU kernels to finish before stopping timer
    torch.cuda.synchronize()

    end_time_gpu = time.time()

    print(f"Time on GPU: {end_time_gpu - start_time_gpu} seconds.")
else:
    print("No CUDA device is available.")
