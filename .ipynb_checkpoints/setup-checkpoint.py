from setuptools import setup, find_packages
from spch_version import __version__

setup(
    name="spchutils",
    version=__version__,
    url="",

    author="Dirk Van Compernolle",
    author_email="compi@esat.kuleuven.be",

    description="A loose collection of speech processing utilities",
    license = "free",
    
    packages = ['spchutils'],
    # a dictionary refering to required data not in .py files
    package_data = {},
    
    install_requires=[
        'numpy >= 1.15.0',
        'scipy >= 1.0.0',
        #'scikit-learn >= 0.14.0, != 0.19.0',
        'numba >= 0.43.0 <= 0.48.0',
        'soundfile >= 0.9.0',
    ],
    
    python_requires='>=3.6',
    
    classifiers=['Development Status: Alpha, Unstable',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8'],
                 
    include_package_data=True

)
