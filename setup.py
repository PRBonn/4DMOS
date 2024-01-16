from setuptools import setup, find_packages

setup(
    name="mos4d",
    version="0.1",
    author="Benedikt Mersch",
    package_dir={"": "src"},
    description="Receding Moving Object Segmentation in 3D LiDAR Data Using Sparse 4D Convolutions",
    packages=find_packages(where="src"),
    install_requires=[
        "Click",
        "numpy",
        "pytorch_lightning",
        "tensorboard",
        "PyYAML",
        "tqdm",
    ],
)
