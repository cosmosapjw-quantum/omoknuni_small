import os
import re
import sys
import platform
import subprocess
import multiprocessing
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


# Get the number of CPU cores for parallel build
cpu_count = max(1, multiprocessing.cpu_count() - 1)

# Read the version from __init__.py
version = '0.1.0'  # Default version
try:
    with open(Path('python/alphazero/__init__.py'), 'r') as f:
        version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read())
        if version_match:
            version = version_match.group(1)
except FileNotFoundError:
    pass


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def run(self):
        # Check if CMake is installed
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the extension"
            )

        # Set build type
        build_type = 'Debug' if self.debug else 'Release'
        
        # Create build directory if it doesn't exist
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure and build
        for ext in self.extensions:
            # Configure
            cmake_args = [
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={str(build_dir)}',
                f'-DPYTHON_EXECUTABLE={sys.executable}',
                f'-DCMAKE_BUILD_TYPE={build_type}',
                '-DBUILD_SHARED_LIBS=ON',
                '-DBUILD_PYTHON_BINDINGS=ON',
                '-DBUILD_TESTS=OFF',
                '-DBUILD_EXAMPLES=OFF',
            ]
            
            # Add platform-specific settings
            if platform.system() == "Windows":
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
            
            # Configure through CMake
            print(f"Configuring CMake with args: {cmake_args}")
            subprocess.check_call(
                ['cmake', ext.cmake_lists_dir] + cmake_args,
                cwd=self.build_temp
            )
            
            # Build
            print("Building extension with CMake")
            build_args = [
                'cmake', '--build', '.', '--config', build_type,
                '--parallel', str(cpu_count),
                '--target', 'alphazero_py', 'alphazero_pipeline'
            ]
            subprocess.check_call(build_args, cwd=self.build_temp)
            
            # Copy extension files to the right location
            self.copy_extensions_to_source()


class CustomInstallCommand(install):
    def run(self):
        # First run the standard install
        install.run(self)
        
        # Then copy any additional files needed


setup(
    name='alphazero',
    version=version,
    author='Omoknuni Team',
    author_email='info@omoknuni.ai',
    description='AlphaZero-Style Multi-Game AI Engine',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/omoknuni/omoknuni',
    license='MIT',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=[
        CMakeExtension('alphazero.alphazero_py'),
        CMakeExtension('alphazero.alphazero_pipeline')
    ],
    cmdclass={
        'build_ext': CMakeBuild,
        'install': CustomInstallCommand,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.10.0',
        'pyyaml>=5.1',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=20.8b1',
            'isort>=5.0.0',
            'flake8>=3.8.0',
        ],
    },
    zip_safe=False,
)