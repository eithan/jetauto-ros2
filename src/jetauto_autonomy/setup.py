from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetauto_autonomy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Eithan',
    maintainer_email='eithan@users.noreply.github.com',
    description='SLAM, navigation, and autonomous exploration for JetAuto',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_explorer = jetauto_autonomy.frontier_explorer:main',
            'safety_monitor = jetauto_autonomy.safety_monitor:main',
        ],
    },
)
