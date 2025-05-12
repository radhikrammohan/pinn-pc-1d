from setuptools import setup, find_packages

setup(
    name="pinn-pc-1d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pde",
        "joblib",
    ],
    author="Radhik Rammohan",
    author_email="radhikrammohan@gmail.com",
    description="A 1D PINN (Physics-Informed Neural Network) package for phase change problems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pinn-pc-1d",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 