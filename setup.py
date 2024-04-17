from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="torchmetrics_ext",
    version="0.1.0",
    description="An extension of torchmetrics package.",
    long_description=long_description,
    author="Yiming Zhang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    keywords=["deep learning", "machine learning", "pytorch", "metrics"],
    python_requires=">=3.8",
    url="https://github.com/eamonn-zh/torchmetrics_ext",
    classifiers=[
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["torchmetrics", "torch"],
    extras_require={"torch": ["torch"]},
)
