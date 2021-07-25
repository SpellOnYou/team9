import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SS2021-Teamlab-group9", 
    version="0.1.0",
    author="",
    author_email="",
    description="Emotion Classification",
    long_description="README.rst",
    url="https://github.com/SpellOnYou/CLab21",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)