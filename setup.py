from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.0',
    author='Ziyan Xiong, Bo Chen',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='multiagent-quadruped-environments',
    install_requires=['isaacgym',
                      'matplotlib',
                      'debugpy']
)