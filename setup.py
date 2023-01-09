from setuptools import setup, find_packages

setup(
    name="triangulation23",
    version='0.0.1',
    install_requires=[
        "numpy",
    ],
    url='https://github.com/ymzkd/DelaunTriangulation2D',
    author='Daisuke Yamazaki',
    packages=find_packages(exclude=('examples', 'test')),
    python_requires='>=3.9',
)
