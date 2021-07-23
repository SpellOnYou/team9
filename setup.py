import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="teamg9", 
    author="",
    author_email="",
    description="NLP Teamlab Group 9 Emotion Classification Library",
    long_description="README.rst",
    url="https://github.com/SpellOnYou/CLab21",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points = {
        'console_scripts': [
            'teamg9 = teamg9.main:main'],
        'distutils.commands': []
    }
)