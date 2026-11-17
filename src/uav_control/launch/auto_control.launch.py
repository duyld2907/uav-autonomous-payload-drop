from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_index = LaunchConfiguration('camera_index')
    frame_width = LaunchConfiguration('frame_width')
    frame_height = LaunchConfiguration('frame_height')
    fps = LaunchConfiguration('fps')
    image_topic = LaunchConfiguration('image_topic')
    fixed_box_w = LaunchConfiguration('fixed_box_w')
    fixed_box_h = LaunchConfiguration('fixed_box_h')
    publish_debug_image = LaunchConfiguration('publish_debug_image')
    save_debug_video = LaunchConfiguration('save_debug_video')
    debug_video_path = LaunchConfiguration('debug_video_path')

    camera_node = Node(
        package='uav_control',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'camera_index': camera_index,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'fps': fps,
            'image_topic': image_topic,
        }],
    )

    vision_node = Node(
        package='uav_control',
        executable='vision_node',
        name='vision_node',
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'fixed_box_w': fixed_box_w,
            'fixed_box_h': fixed_box_h,
            'publish_debug_image': publish_debug_image,
            'save_debug_video': save_debug_video,
            'debug_video_path': debug_video_path,
        }],
    )

    auto_control_node = Node(
        package='uav_control',
        executable='auto_control_node',
        name='auto_control_node',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('frame_width', default_value='640'),
        DeclareLaunchArgument('frame_height', default_value='480'),
        DeclareLaunchArgument('fps', default_value='20.0'),
        DeclareLaunchArgument('image_topic', default_value='/cam/front/image_raw'),
        DeclareLaunchArgument('fixed_box_w', default_value='120'),
        DeclareLaunchArgument('fixed_box_h', default_value='120'),
        DeclareLaunchArgument('publish_debug_image', default_value='false'),
        DeclareLaunchArgument('save_debug_video', default_value='true'),
        DeclareLaunchArgument('debug_video_path', default_value='/ws/logs/vision_runs/vision_debug_launch.avi'),
        camera_node,
        vision_node,
        auto_control_node,
    ])