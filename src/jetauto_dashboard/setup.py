import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'jetauto_dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/static', glob('static/*')),
    ],
    install_requires=[
        'setuptools',
        'flask',
        'flask-socketio',
        'simple-websocket',
    ],
    zip_safe=True,
    maintainer='Eithan',
    maintainer_email='eithan@users.noreply.github.com',
    description='Full-screen control panel dashboard for JetAuto robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard_node = jetauto_dashboard.dashboard_node:main',
        ],
    },
)
