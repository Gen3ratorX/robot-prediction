from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = Path(get_package_share_directory('robot_prediction'))
    default_world = pkg_share / 'worlds' / 'plain_field.world'

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=str(default_world),
        description='Plain Gazebo world file',
    )

    gazebo = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-r',
            LaunchConfiguration('world'),
        ],
        output='screen',
    )

    return LaunchDescription([
        world_arg,
        gazebo,
    ])
