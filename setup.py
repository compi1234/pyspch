from setuptools import setup, find_packages   
from spch_version import __version__   

setup(
    name="pyspch",
    version=__version__,
    url="",

    author="Dirk Van Compernolle",
    author_email="compi@esat.kuleuven.be",

    description="A loose collection of speech processing utilities",
    license = "free",
    
    packages = ['pyspch'],
    # add spch_version to the required install modules
    py_modules = ['spch_version'],
    # a dictionary refering to required data not in .py files
    package_data = {},
    
    install_requires=[
        'numpy >= 1.15.0',
        'scipy >= 1.0.0',
        'pandas >= 1.0',
        'librosa == 0.8.0',
        'numba >= 0.43.0,<= 0.48.0',
        'soundfile >= 0.9.0',
        'matplotlib >= 3.1',
        'ipython >= 7.21'
        'ipywidgets >= 7.5.1',
        'plotly >= 4.12',
        'pydub'
    ],
    
    python_requires='>=3.6',
    
    classifiers=['Development Status: Alpha, Unstable',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8'],
                 
    include_package_data=True

)
