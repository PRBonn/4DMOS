from setuptools import setup, find_packages

setup(
    name="mos4d",
    version="0.1",
    author="Benedikt Mersch",
    package_dir={"": "src"},
    description="Receding Moving Object Segmentation in 3D LiDAR Data Using Sparse 4D Convolutions",
    packages=find_packages(where="src"),
    install_requires=[
        "Click>=7.0",
        "numpy>=1.20.3",
        "pytorch_lightning>=1.6.4",
        "PyYAML>=6.0",
        "tqdm>=4.62.3",
        "torch",
        "ninja",
    ],
)
