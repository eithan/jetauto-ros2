from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetauto_faces'

setup(
    name=package_name,
    version='0.1.0',
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
    description='Face recognition for JetAuto using InsightFace',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_recognition_node = jetauto_faces.face_recognition_node:main',
            'enroll_face = jetauto_faces.enroll_face:main',
            'enrollment_node = jetauto_faces.enrollment_node:main',
        ],
    },
)
