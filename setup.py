#!/usr/bin/env python

"""The setup script."""

from pathlib import Path

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


# Read the requirements
source_root = Path(".")
with (source_root / "requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()

test_requirements = [
    "pytest>=3",
]

setup(
    author="Alexandra Kapp",
    author_email="alexandra.kapp@htw-berlin.de",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="Create a report for mobility data with differential privacy guarantees.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="dp_mobility_report",
    name="dp-mobility-report",
    packages=find_packages(include=["dp_mobility_report", "dp_mobility_report.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/FreeMoveProject/dp_mobility_report",
    version="0.0.4",
    zip_safe=False,
)
