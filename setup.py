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
    
    packages = ['pyspch','spchdata'],
    # add spch_version to the required install modules
    py_modules = ['spch_version'],
    # a dictionary refering to required data not in .py files
    # include_package_data=True  (this would look into MANIFEST.in
    #package_data = {'pyspch':['my_data/*']},
    package_data = {'spchdata':['*']},
    
    install_requires=[
        'numpy >= 1.19.0',
        'scipy >= 1.4.0',
        'pandas >= 1.1',
        'librosa >= 0.8.0',
        'numba',
        'soundfile >= 0.10.0',
        'matplotlib >= 3.2',
        'ipywidgets >= 7.5.1',
        'scikit-learn >= 1.0.0',
        'pydub'
    ],
    
    python_requires='>=3.7',
    
    classifiers=['Development Status: Beta, Unstable',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9'],
                 


)
