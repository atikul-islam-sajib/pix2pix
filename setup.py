from setuptools import setup, find_packages

def requirements():
    with open("./requirements.txt", "r") as file:
        return file.read().splitlines()

setup(
    name="pix2pix",
    version="0.1.0",
    description="A deep learning project that is build for GAN based image to image translation for the Image dataset",
    author="Atikul Islam Sajib",
    author_email="atikul.sajib@ptb.de",
    url="https://github.com/atikul-islam-sajib/pix2pix.git",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    install_requires = requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="pix2pix machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/pix2pix.git/issues",
        "Documentation": "https://atikul-islam-sajib.github.io/pix2pix-deploy/",
        "Source Code": "https://github.com/atikul-islam-sajib/pix2pix.git",
    },
)