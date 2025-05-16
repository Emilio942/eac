"""
Setup file for the Emergent Adaptive Core (EAC) package.
"""

from setuptools import setup, find_packages

setup(
    name="eac",
    version="0.1.0",
    description="Emergent Adaptive Core (EAC) - A Self-Evolving AI Architecture",
    author="Emilio",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "tensorflow>=2.8.0",
        "torch>=1.11.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
        "pandas>=1.4.2",
        "pytest>=7.0.0",
        "networkx>=2.7.1",
        "gymnasium>=0.26.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
