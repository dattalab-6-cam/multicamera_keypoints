from setuptools import find_packages, setup

setup(
    name="multicamera_keypoints",
    packages=find_packages(),
    version="0.1.0",
    description="A pipeline to go from 6-cam videos to usable keypoints",
    install_requires=[
        "av",
        "numpy",
        "opencv-python",
        "pandas",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
        "torch",  # TODO: figure out how to get the right torch version upon pip install? or just do it second?
        "scipy",
        "vidio",
        "tqdm",
        "matplotlib",
    ],
    author="dattalab",
    license="MIT",
)
