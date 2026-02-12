from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetauto_tts'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Eithan',
    maintainer_email='eithan@users.noreply.github.com',
    description='Text-to-speech for JetAuto — announces detected objects',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tts_node = jetauto_tts.tts_node:main',
        ],
    },
)
