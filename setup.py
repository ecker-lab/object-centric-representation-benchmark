from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    "numpy",
    "motmetrics==1.2.0",
    "torch==1.5.0",
    "torchvision==0.6.0"
]


setup(name='ocrb', 
        version='1.0',
        description="Object-centric Representation Benchmark (OCRB) contains code, data and a benchmark leaderboard from the paper 'Unmasking the Inductive Biases of Unsupervised Object Representations for Video Sequences'",
        author="Marissa Weis",
        author_email="marissa.weis@bethgelab.org",
        url="https://eckerlab.org/code/weis2020/",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3"
                 ],
        packages=find_packages(),
        install_requires=install_requires,
        )
