"""Setup for federated-tinyml-vessel: the F170 federated on-device learning system."""
from setuptools import setup, find_packages

setup(
    name="federated-tinyml-vessel",
    version="0.1.0",
    description="F170 — Federated TinyML for the Vessel Edge. Frozen audio backbone + 1.3 KB classifier head + FedAvg.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Patrick McNamara",
    author_email="Mavis@superinstance.dev",
    url="https://github.com/SuperInstance/federated-tinyml-vessel",
    py_modules=["feature_extractor", "classifier_head", "simulator", "federated", "study"],
    install_requires=["numpy"],
    python_requires=">=3.9",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
