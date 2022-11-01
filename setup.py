from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    version                       = "0.1dev",
    name                          = "RidgePy",
    description                   = "Kalman Filter for audio signals.",
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    packages                      = ["ridgepy"],
    install_requires              = [
        "numpy",
        "matplotlib"
    ],
    include_package_date          = True,
    scripts                       = [
        "bin/simulation/0.0-simulate-single-mode.py",
        "bin/simulation/0.1-simulate-multi-mode.py",
    ],
)