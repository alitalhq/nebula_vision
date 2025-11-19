from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'nebula_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'calib'),  glob('calib/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alitalhq',
    maintainer_email='alitqlhq@gmail.com',
    description='Nebula Vision nodes (camera driver, vision processor).',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
