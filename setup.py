import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()


setuptools.setup(
    name="tsms",
    version="0.0.1",
    author="Fabio Di Marco",
    author_email="fabiodimarco@hotmail.it",
    description="Tensorflow sound analysis / synthesis library for musical applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabiodimarco/tf-spectral-modeling-synthesis.git",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
