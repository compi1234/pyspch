from setuptools import setup, find_packages
from version import __version__
    
setup(
    name="spchutils",
    version=__version__,
    url="",

    author="Dirk Van Compernolle",
    author_email="compi@esat.kuleuven.be",

    description="A loose collection of speech processing utilities",
    license = "free",
    
    # packages = ['spchutils'],
    packages = find_packages(),
    # a dictionary refering to required data not in .py files
    package_data = {},
    
    install_requires=['numpy','pandas','matplotlib'],

    classifiers=['Development Status: Alpha, Unstable',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8'],
                 
    include_package_data=True

)