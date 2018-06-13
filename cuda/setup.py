from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spacc_cuda',
    ext_modules=[
        CUDAExtension('spacc_cuda', [
            'spacc_cuda.cpp',
            'spacc_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
