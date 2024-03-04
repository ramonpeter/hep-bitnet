from setuptools import setup, find_packages

HTTPS_GITHUB_URL = "https://github.com/ramonpeter/hep-bitnet"

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy", "pandas", "scipy", "tables", "torch", "torchtestcase", "matplotlib", "vegas"]

setup(
    name="bithep",
    version="2.0.0",
    author="Claudius Krause, Daohan Wang, Ramon Winterhalder",
    author_email="ramon.winterhalder@uclouvain.be",
    description="1-Bit networks for HEP applications",
    long_description=long_description,
    long_description_content_type="text/md",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=requirements,
)
