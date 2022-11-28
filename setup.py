import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="1to7_ecg_generator", # Replace with your own username
    version="0.0.1",
    author="Tobias Willi",
    author_email="twilli.gnl@gmail.com",
    description="translates lead 1 ECG to 12 lead ECG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simulamet-host/ecg_1_to_8/tree/side",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'deepfakeecg': ['checkpoints/g_stat.pt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tqdm',
        'pandas',
        "torch",
  ],
)