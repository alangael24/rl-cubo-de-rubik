"""
setup.py - Build script for Rubik's Cube C extension

Usage:
    python setup.py build_ext --inplace
    pip install -e .
"""

from setuptools import setup, Extension
import numpy as np
import sys
import os

# Get numpy include directory
numpy_include = np.get_include()

# Compiler flags for optimization
extra_compile_args = ['-O3', '-ffast-math', '-march=native']
if sys.platform == 'darwin':  # macOS
    extra_compile_args.append('-stdlib=libc++')

# Debug mode
if os.environ.get('DEBUG'):
    extra_compile_args = ['-g', '-O0', '-fsanitize=address']

# Define the extension module
rubik_c_extension = Extension(
    'rubik_c',
    sources=['csrc/rubik_binding.c'],
    include_dirs=[numpy_include, 'csrc'],
    extra_compile_args=extra_compile_args,
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)

setup(
    name='rubik-cube-rl',
    version='1.0.0',
    description='High-performance Rubik\'s Cube RL environment',
    author='Claude',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'gymnasium>=0.29.0',
    ],
    extras_require={
        'train': [
            'torch>=2.0.0',
            'pufferlib>=3.0.0',
        ],
    },
    ext_modules=[rubik_c_extension],
    py_modules=['rubik_env_c', 'train'],
)
