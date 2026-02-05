"""Setup configuration for weex-sdk package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="weex-sdk",
    version="1.0.10",
    author="Weex SDK Contributors",
    description="A comprehensive Python SDK for Weex exchange API (HTTP + WebSocket)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weex/weex-sdk-python",
    packages=find_packages(),
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
    },
)
