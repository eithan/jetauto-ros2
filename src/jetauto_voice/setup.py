from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetauto_voice'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Eithan',
    maintainer_email='eithan@users.noreply.github.com',
    description='Custom voice control for JetAuto',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_control_node = jetauto_voice.voice_control_node:main',
            'voice_commander_node = jetauto_voice.voice_commander_node:main',
        ],
    },
)
