from setuptools import find_packages, setup


package_name = 'robot_prediction'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/view_robot.launch.py']),
        (f'share/{package_name}/urdf', ['urdf/robot_prediction_demo.urdf']),
        (f'share/{package_name}/rviz', ['rviz/robot_prediction_demo.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='st.dominic',
    maintainer_email='st.dominic@example.com',
    description='Human-aware robot navigation with gesture, movement, and action prediction.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'combined_ros2_node = robot_prediction.combined_ros2_node:main',
            'ros_inference_node = robot_prediction.ros_inference_node:main',
        ],
    },
)
