from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'position_estimator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/position_estimator.launch.py']),
        ('share/' + package_name + '/models', glob('models/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Eojin Kang',
    maintainer_email='ejyl0818@gmail.com',
    description='Position estimation using ray-plane calculation and ConvNeXt model',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'position_estimator_node = position_estimator.position_estimator_node:main',
        ],
    },
)
