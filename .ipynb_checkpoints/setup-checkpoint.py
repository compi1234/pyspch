from setuptools import setup, find_packages   
from spch_version import __version__   

setup(
    name="pyspch",
    version=__version__,
    url="",

    author="Dirk Van Compernolle",
    author_email="dirk.vancompernolle@kuleuven.be",

    description="A loose collection of speech processing utilities",
    license = "free",
    
    packages = find_packages(),
    # add spch_version to the required install modules
    py_modules = ['spch_version'],
    
    # a dictionary referring to required data not in .py files
    include_package_data=True  
    package_data = {'pyspch':['data/*','data/demo/*']},
    
    install_requires=[
        'numpy >= 1.22',
        'scipy >= 1.10 ',
        'pandas >= 1.3 ',
        'librosa >= 0.10',
        'soundfile >= 0.11.0',
        'matplotlib >= 3.5, < 4',
        'ipywidgets >= 7.7',
        'scikit-learn >= 1.2.0',
        'seaborn >= 0.12',
        'pydub>=0.23,<0.26'
    ],
    
    python_requires='>=3.8',
    
    classifiers=['Development Status: Functional, Beta',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.8'],
                 


)
