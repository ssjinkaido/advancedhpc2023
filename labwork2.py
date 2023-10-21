from numba import cuda

print(f"Gpu info: {cuda.detect()}")
print(f"Cuda device: {cuda.devices}")
print(f"Cuda gpus: {cuda.gpus}")
print(f"Device choosen: {cuda.select_device(0)}")
multiprocessor_count = cuda.gpus[0].MULTIPROCESSOR_COUNT
print(f"Multiprocessor Count: {multiprocessor_count}")

gpu_memory_size = cuda.current_context().get_memory_info()
print(f"GPU Memory Size: {gpu_memory_size.total // (1024**2)} MB")
