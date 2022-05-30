from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='os_riroi_align_cuda',
    ext_modules=[
        CUDAExtension('os_riroi_align_cuda', [
            'src/os_riroi_align_cuda.cpp',
            'src/os_riroi_align_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
