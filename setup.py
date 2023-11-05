import os
from setuptools import setup
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
   README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='CountTrees',
    version='1.0a0',
    packages=find_packages(),

    download_url='',

    install_requires=[ 
                     'DeepForest',
                    'click',
                    'opencv-python-headless',
                    'albumentations',
                    'docopt',
                    'GDAL',
                    'tqdm',
                    'geopandas',
    ],

    entry_points={'console_scripts': [
        'inference=count_trees.inference:main',
        'train=count_trees.train:main',
        'generate_dataset=count_trees.generate_dataset:main',
        'processing_data=count_trees.utils.processing_data:main'
        ]
        },

    include_package_data=True,
    #license='MIT License',
    description="",
    zip_safe=False,

    long_description=README,
    long_description_content_type='text/markdown',

    python_requires='>=3.10',

)