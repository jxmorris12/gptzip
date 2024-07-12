from setuptools import find_packages, setup

setup(
    name="gptzip",
    version="0.0.1",
    description="losslessly compress data with language models",
    author="Jack Morris",
    author_email="jxm3@cornell.edu",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
)
