from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="equipment-maintenance-pipeline",
    version="0.1.0",
    author="Shinly",
    description="NLP pipeline for industrial equipment maintenance — topic classification, embedding clustering, and FLAN-T5 summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shinly/helparooni",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Industrial :: Maintenance",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "maintenance-pipeline=src.pipeline:main",
        ],
    },
)
