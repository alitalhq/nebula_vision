from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_profiles_file = PathJoinSubstitution([
        FindPackageShare('nebula_vision'),
        'config',
        'camera_profiles.yaml'
    ])

    profiles_file_arg = DeclareLaunchArgument(
        'profiles_file',
        default_value=default_profiles_file,
        description='Profil YAML dosyasının yolu (camera_profiles.yaml)'
    )

    profile_arg = DeclareLaunchArgument(
        'profile',
        default_value='internal',
        description='Kullanılacak kamera profili (ör. internal, front, down)'
    )

    node = Node(
        package='nebula_vision',
        executable='camera_driver',
        name='camera_driver',
        output='screen',
        parameters=[{
            'profiles_file': LaunchConfiguration('profiles_file'),
            'profile': LaunchConfiguration('profile'),
        }]
    )

    return LaunchDescription([
        profiles_file_arg,
        profile_arg,
        node
    ])
