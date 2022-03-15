from setuptools import setup, find_packages

setup(
    name="triangulation",
    version='0.0.1',
    install_requires=[
        "numpy",
    ],
    author='Daisuke Yamazaki',
    packages=find_packages(exclude=('examples')),
    python_requires='>=3.9',
)
