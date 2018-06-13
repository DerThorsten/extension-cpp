from torch.utils.cpp_extension import load
spacc_cuda = load(
    'spacc_cuda', ['spacc_cuda.cpp', 'spacc_cuda_kernel.cu'], verbose=True)
help(spacc_cuda)
