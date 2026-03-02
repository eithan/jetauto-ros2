import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'jetauto_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['manifest.json']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'images'), glob('images/*.jpg')),
        (os.path.join('share', package_name, 'images'), glob('images/*.jpeg')),
        (os.path.join('share', package_name, 'images'), glob('images/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jarvis',
    maintainer_email='jarvis@openclaw.ai',
    description='Simulation tools for JetAuto — fake image publisher for offline testing.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_node = jetauto_sim.sim_node:main',
        ],
    },
)
