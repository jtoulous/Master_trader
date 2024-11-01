from setuptools import setup, find_packages

setup(
    name='utils',
    version='0.1',
    packages=['utils'],
    package_dir={'': 'core'},
    install_requires=[
        'colorama',
        'pandas',
        'numpy',
        'scipy',
        'scikit-learn',
    ],
)