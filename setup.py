import setuptools
from io import open

setuptools.setup(
    name="team9", 
    version="1.7",
    author="Jiwon Kim and Lara Grimminger",
    author_email="st176776@stud.uni-stuttgart.de", "st157146@stud.uni-stuttgart.de",
    description="NLP Teamlab Group 9 Emotion Classification Library",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SpellOnYou/CLab21",
    packages=setuptools.find_packages(exclude=["experimental"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas>1.0',
        'numpy'>=1.15,
        'tensorflow>=2.0',
        'tensorflow-addons>=0.11',
        'numpy',
        'sklearn',
        'lime',
        'seaborn'
    ],
    python_requires='>=3.6',
    include_package_data=True,
    entry_points = {
        'console_scripts': [
            'team9-emo-cls=team9_cli.main:main',
        ]
    }
)

