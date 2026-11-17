from setuptools import setup
from glob import glob
import os

package_name = 'uav_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@example.com',
    description='UAV control package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = uav_control.camera_node:main',
            'vision_node = uav_control.vision_node:main',
            'auto_control_node = uav_control.auto_control_node:main',
            'testservo = uav_control.testservo:main',
            'servo = uav_control.servo:main',
        ],
    },
)