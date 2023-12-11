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
    
    packages = find_packages(),
    # add spch_version to the required install modules
    py_modules = ['spch_version'],
    # a dictionary referring to required data not in .py files
    
    # include_package_data=True  
    package_data = {'pyspch':['data/*']},
    
    install_requires=[
        'numpy >= 1.19.0, <1.22',
        'scipy >= 1.4.0 ',
        'pandas >= 1.3 ',
        'librosa >= 0.8.1, <0.9',
        'soundfile >= 0.10.0',
        'matplotlib >= 3.4, < 4',
        'ipywidgets >= 7.5.1, < 9',
        'scikit-learn >= 1.0.0',
        'seaborn >= 0.11',
        'pydub>=0.23,<0.26'
    ],
    
    python_requires='>=3.7',
    
    classifiers=['Development Status: Functional, Beta',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9'],
                 


)
