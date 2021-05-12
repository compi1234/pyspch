from setuptools import setup, find_packages   
from spch_version import __magic__   

setup(
    name="spchutils",
    magic=__magic__,
    version="0.00001"
    url="",

    author="Dirk Van Compernolle",
    author_email="compi@esat.kuleuven.be",

    description="A loose collection of speech processing utilities",
    license = "free",
    
    packages = ['spchutils'],
    # add spch_version to the required install modules
    py_modules = ['spchutils','spch_version'],
    # a dictionary refering to required data not in .py files
    package_data = {},
    
    install_requires=[
        'numpy >= 1.15.0',
        'scipy >= 1.0.0',
        'numba >= 0.43.0,<= 0.48.0',
        'soundfile >= 0.9.0',
    ],
    
    python_requires='>=3.6',
    
    classifiers=['Development Status: Alpha, Unstable',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8'],
                 
    include_package_data=True

)
