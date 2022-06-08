#!/usr/bin/env python

import threading
import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
from smach import State,StateMachine
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray ,PointStamped
from std_msgs.msg import Empty
from tf import TransformListener
import tf
import math
import numpy as np
import rospkg
import csv
import time
from geometry_msgs.msg import PoseStamped
import follow_waypoints.msg
from follow_waypoints.msg import WaypointsAction, WaypointsResult, WaypointsFeedback

from tf.transformations import quaternion_from_matrix as matrix2quaternion
from tf.transformations import unit_vector as normalize_vector


def outer_product_matrix(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def cross_product(a, b):
    return np.dot(outer_product_matrix(a), b)


def rotation_matrix_from_axis(
        first_axis=(1, 0, 0), second_axis=(0, 1, 0), axes='xy'):
    if axes not in ['xy', 'yx', 'xz', 'zx', 'yz', 'zy']:
        raise ValueError("Valid axes are 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'.")
    e1 = normalize_vector(first_axis)
    e2 = normalize_vector(second_axis - np.dot(second_axis, e1) * e1)
    if axes in ['xy', 'zx', 'yz']:
        third_axis = cross_product(e1, e2)
    else:
        third_axis = cross_product(e2, e1)
    e3 = normalize_vector(
        third_axis - np.dot(third_axis, e1) * e1 - np.dot(third_axis, e2) * e2)
    first_index = ord(axes[0]) - ord('x')
    second_index = ord(axes[1]) - ord('x')
    third_index = ((first_index + 1) ^ (second_index + 1)) - 1
    indices = [first_index, second_index, third_index]
    return np.vstack([e1, e2, e3])[np.argsort(indices)].T

# change Pose to the correct frame 
def changePose(waypoint,target_frame):
    if waypoint.header.frame_id == target_frame:
        # already in correct frame
        return waypoint
    if not hasattr(changePose, 'listener'):
        changePose.listener = tf.TransformListener()
    tmp = PoseStamped()
    tmp.header.frame_id = waypoint.header.frame_id
    tmp.pose = waypoint.pose.pose
    try:
        changePose.listener.waitForTransform(
            target_frame, tmp.header.frame_id, rospy.Time(0), rospy.Duration(3.0))
        pose = changePose.listener.transformPose(target_frame, tmp)
        ret = PoseWithCovarianceStamped()
        ret.header.frame_id = target_frame
        ret.pose.pose = pose.pose
        return ret
    except:
        rospy.loginfo("CAN'T TRANSFORM POSE TO {} FRAME".format(target_frame))
        exit()

def changeOrientaion(i):
    global waypoints
    if i == len(waypoints)-1:
        return
    pose_a = waypoints[i].pose.pose
    pose_b = waypoints[i+1].pose.pose
    pos_a = np.array([pose_a.position.x, pose_a.position.y, pose_a.position.z])
    pos_b = np.array([pose_b.position.x, pose_b.position.y, pose_b.position.z])
    v_ab = pos_b - pos_a
    if np.linalg.norm(v_ab) == 0.0:
        return
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix_from_axis(v_ab, (0, 0, 1), axes="xz")
    q_xyzw = matrix2quaternion(matrix)
    waypoints[i].pose.pose.orientation.x = q_xyzw[0]
    waypoints[i].pose.pose.orientation.y = q_xyzw[1]
    waypoints[i].pose.pose.orientation.z = q_xyzw[2]
    waypoints[i].pose.pose.orientation.w = q_xyzw[3]


def changeOrientaionModified(waypoints):
    n = len(waypoints)
    for i in range(n - 1):
        pose_a = waypoints[i].pose.pose
        pose_b = waypoints[i+1].pose.pose
        pos_a = np.array([pose_a.position.x, pose_a.position.y, pose_a.position.z])
        pos_b = np.array([pose_b.position.x, pose_b.position.y, pose_b.position.z])
        v_ab = pos_b - pos_a
        if np.linalg.norm(v_ab) == 0.0:
            continue
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix_from_axis(v_ab, axes="xz")
        q_xyzw = matrix2quaternion(matrix)
        print(q_xyzw)
        waypoints[i].pose.pose.orientation.x = q_xyzw[0]
        waypoints[i].pose.pose.orientation.y = q_xyzw[1]
        waypoints[i].pose.pose.orientation.z = q_xyzw[2]
        waypoints[i].pose.pose.orientation.w = q_xyzw[3]

#Path for saving and retreiving the pose.csv file 
output_file_path = rospkg.RosPack().get_path('follow_waypoints')+"/saved_path/pose.csv"
waypoints = []

class FollowPath(State):
    global fw
    def __init__(self):
        State.__init__(self, outcomes=['success'], input_keys=['waypoints'])
        self.frame_id = rospy.get_param('~goal_frame_id','map')
        self.odom_frame_id = rospy.get_param('~odom_frame_id','odom')
        self.base_frame_id = rospy.get_param('~base_frame_id','base_footprint')
        self.duration = rospy.get_param('~wait_duration', 0.0)
        # Get a move_base action client
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('Connecting to move_base...')
        self.client.wait_for_server()
        rospy.loginfo('Connected to move_base.')
        rospy.loginfo('Starting a tf listner.')
        self.tf = TransformListener()
        self.listener = tf.TransformListener()
        self.distance_tolerance = rospy.get_param('~waypoint_distance_tolerance', 0.0)
        self.timeout = rospy.get_param('~timeout', 0.0)

    def execute(self, userdata):
        global waypoints
        # Execute waypoints each in sequence
        rospy.loginfo('waypoints:{}'.format(len(waypoints)))
        for i, waypoint in enumerate(waypoints):
            # print(waypoint)
            # Break if preempted
            if waypoints == []:
                fw.set_result(WaypointsResult.RESET)
                rospy.loginfo('The waypoint queue has been reset.')
                return 'success'
            elif fw._as.is_preempt_requested():
                self.client.cancel_all_goals()
                fw.set_result(WaypointsResult.CANCELED)
                rospy.loginfo('Waypoints action Server received cancel request.')
                return 'success'

            # Otherwise publish next waypoint as goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = self.frame_id
            goal.target_pose.pose.position = waypoint.pose.pose.position
            goal.target_pose.pose.orientation = waypoint.pose.pose.orientation
            rospy.loginfo('Executing move_base goal to position (x,y): %s, %s' %
                    (waypoint.pose.pose.position.x, waypoint.pose.pose.position.y))
            # rospy.loginfo("To cancel the goal: 'rostopic pub -1 /move_base/cancel actionlib_msgs/GoalID -- {}'")
            self.client.send_goal(goal)
            if not self.distance_tolerance > 0.0:
                if self.timeout > 0:
                    rospy.loginfo('Set timeout:{}'.format(self.timeout))
                    finished_within_time = self.client.wait_for_result(rospy.Duration(self.timeout))
                    if not finished_within_time:
                        rospy.loginfo('Cancelled move_base because of timeout {}s'.format(self.timeout))
                        self.client.cancel_all_goals()
                        fw.set_result(WaypointsResult.TIMEOUT)
                        return 'success'
                else:
                    self.client.wait_for_result()
                state = self.client.get_state()
                if state == GoalStatus.ABORTED or state == GoalStatus.PREEMPTED:
                    rospy.loginfo('Move_base failed because server received cancel request or goal was aborted')
                    fw.set_result(WaypointsResult.FAILED)
                    return 'success'
                if self.duration > 0.0:
                    rospy.loginfo("Waiting for %f sec..." % self.duration)
                    time.sleep(self.duration)
                fw.send_feedback("Passed {}/{} waypoints.".format(i+1, len(waypoints)))
            else:
                #This is the loop which exist when the robot is near a certain GOAL point.
                distance = 10
                start_time = rospy.Time.now()
                if self.timeout > 0:
                    rospy.loginfo('Set timeout:{}'.format(self.timeout))
                    timeout = rospy.Duration(secs=self.timeout)
                else:
                    self.listener.waitForTransform(self.frame_id, self.base_frame_id, start_time, rospy.Duration(4.0))
                    trans,rot = self.listener.lookupTransform(self.frame_id, self.base_frame_id, start_time)
                    calculated_timeout = max(10.0, 10 * math.sqrt(pow(waypoint.pose.pose.position.x-trans[0],2)+pow(waypoint.pose.pose.position.y-trans[1],2)))
                    rospy.loginfo('Set timeout:{}'.format(calculated_timeout))
                    timeout = rospy.Duration(secs=calculated_timeout)

                while(distance > self.distance_tolerance):
                    now = rospy.Time.now()
                    self.listener.waitForTransform(self.frame_id, self.base_frame_id, start_time, rospy.Duration(4.0))
                    trans,rot = self.listener.lookupTransform(self.frame_id, self.base_frame_id, start_time)
                    distance = math.sqrt(pow(waypoint.pose.pose.position.x-trans[0],2)+pow(waypoint.pose.pose.position.y-trans[1],2))
                    state = self.client.get_state()
                    if state == GoalStatus.ABORTED or state == GoalStatus.PREEMPTED:
                        rospy.loginfo('Move_base failed because server received cancel request or goal was aborted')
                        fw.set_result(WaypointsResult.FAILED)
                        return 'success'
                    elif (now - start_time) > timeout:
                        rospy.loginfo('Cancelled move_base because of timeout {}s'.format(timeout))
                        self.client.cancel_all_goals()
                        fw.set_result(WaypointsResult.TIMEOUT)
                        return 'success'
                    fw.send_feedback("[{}/{}] {}m to next waypoint.".format(i+1, len(waypoints), distance))
        rospy.loginfo('Reached final waypoint')
        fw.set_result(WaypointsResult.SUCCEEDED)
        return 'success'

def convert_PoseWithCovArray_to_PoseArray(waypoints):
    """Used to publish waypoints as pose array so that you can see them in rviz, etc."""
    poses = PoseArray()
    poses.header.frame_id = rospy.get_param('~goal_frame_id','map')
    poses.poses = [pose.pose.pose for pose in waypoints]
    return poses

class GetPath(State):
    global fw
    def __init__(self):
        State.__init__(self, outcomes=['success'], input_keys=['waypoints'], output_keys=['waypoints'])
        self.frame_id = rospy.get_param('~goal_frame_id','map')
        # Subscribe to pose message to get new waypoints
        self.addpose_topic = rospy.get_param('~addpose_topic','/initialpose')
        # Create publsher to publish waypoints as pose array so that you can see them in rviz, etc.
        self.posearray_topic = rospy.get_param('~posearray_topic','/waypoints')
        self.poseArray_publisher = rospy.Publisher(self.posearray_topic, PoseArray, queue_size=1)

        # Start thread to listen for reset messages to clear the waypoint queue
        def wait_for_path_reset():
            """thread worker function"""
            global waypoints
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                data = rospy.wait_for_message('/path_reset', Empty)
                rospy.loginfo('Recieved path RESET message')
                self.initialize_path_queue()
                rate.sleep()
                # rospy.sleep(3) # Wait 3 seconds because `rostopic echo` latches
                               # for three seconds and wait_for_message() in a
                               # loop will see it again.
        reset_thread = threading.Thread(target=wait_for_path_reset)
        reset_thread.daemon = True  # terminate when main thread exit
        reset_thread.start()

    def initialize_path_queue(self):
        global waypoints
        waypoints = [] # the waypoint queue
        # publish empty waypoint queue as pose array so that you can see them the change in rviz, etc.
        self.poseArray_publisher.publish(convert_PoseWithCovArray_to_PoseArray(waypoints))

    def execute(self, userdata):
        global waypoints
        self.initialize_path_queue()
        self.path_ready = False
        fw.path_ready = False

        # Start thread to listen for when the path is ready (this function will end then)
        # Also will save the clicked path to pose.csv file
        def wait_for_path_ready():
            """thread worker function"""
            data = rospy.wait_for_message('/path_ready', Empty)
            rospy.loginfo('Recieved path READY message')
            self.path_ready = True
            with open(output_file_path, 'w') as file:
                for current_pose in waypoints:
                    file.write(str(current_pose.pose.pose.position.x) + ',' + str(current_pose.pose.pose.position.y) + ',' + str(current_pose.pose.pose.position.z) + ',' + str(current_pose.pose.pose.orientation.x) + ',' + str(current_pose.pose.pose.orientation.y) + ',' + str(current_pose.pose.pose.orientation.z) + ',' + str(current_pose.pose.pose.orientation.w)+ '\n')
            rospy.loginfo('poses written to '+ output_file_path)
        ready_thread = threading.Thread(target=wait_for_path_ready)
        ready_thread.daemon = True  # terminate when main thread exit
        ready_thread.start()

        self.start_journey_bool = False

        # Start thread to listen start_jorney 
        # for loading the saved poses from follow_waypoints/saved_path/poses.csv
        def wait_for_start_journey():
            """thread worker function"""
            data_from_start_journey = rospy.wait_for_message('start_journey', Empty)
            rospy.loginfo('Recieved path READY start_journey')
            with open(output_file_path, 'r') as file:
                reader = csv.reader(file, delimiter = ',')
                for row in reader:
                    print (row)
                    current_pose = PoseWithCovarianceStamped() 
                    current_pose.pose.pose.position.x     =    float(row[0])
                    current_pose.pose.pose.position.y     =    float(row[1])
                    current_pose.pose.pose.position.z     =    float(row[2])
                    current_pose.pose.pose.orientation.x = float(row[3])
                    current_pose.pose.pose.orientation.y = float(row[4])
                    current_pose.pose.pose.orientation.z = float(row[5])
                    current_pose.pose.pose.orientation.w = float(row[6])
                    waypoints.append(current_pose)
                    self.poseArray_publisher.publish(convert_PoseWithCovArray_to_PoseArray(waypoints))
            self.start_journey_bool = True
            
        start_journey_thread = threading.Thread(target=wait_for_start_journey)
        start_journey_thread.daemon = True  # terminate when main thread exit
        start_journey_thread.start()

        topic = self.addpose_topic;
        rospy.loginfo("Waiting to recieve waypoints via Pose msg on topic %s" % topic)
        # rospy.loginfo("To start following waypoints: 'rostopic pub /path_ready std_msgs/Empty -1'")
        # rospy.loginfo("or 'rostopic pub %s/goal follow_waypoints/WaipointsGoal -1'" % rospy.get_param('~action_name','waypoints_action'))
        # rospy.loginfo("OR")
        # rospy.loginfo("To start following saved waypoints: 'rostopic pub /start_journey std_msgs/Empty -1'")

        # Wait for published waypoints or saved path loaded
        pose_sub = rospy.Subscriber(topic, PoseWithCovarianceStamped,
                                    callback=self.callback, queue_size=1000)
        rate = rospy.Rate(10)
        while (not self.path_ready and not self.start_journey_bool and not fw.path_ready):
            rate.sleep()
        pose_sub.unregister()
        # Path is ready! return success and move on to the next state (FOLLOW_PATH)
        # rospy.sleep(1)
        return 'success'

    def callback(self, msg):
        global waypoints
        rospy.loginfo("Recieved new waypoint")
        waypoints.append(changePose(msg, self.frame_id))
        changeOrientaion(len(waypoints)-2)
        # publish waypoint queue as pose array so that you can see them in rviz, etc.
        self.poseArray_publisher.publish(convert_PoseWithCovArray_to_PoseArray(waypoints))

    # def publish_waypoints(self):
    #     global waypoints
    #     rate = rospy.Rate(10)
    #     num = len(waypoints)
    #     while not rospy.is_shutdown():
    #         if num < len(waypoints):
    #             self.poseArray_publisher.publish(convert_PoseWithCovArray_to_PoseArray(waypoints))
    #             num = len(waypoints)
    #         rate.sleep()

    # publish_thread = threading.Thread(target=publish_waypoints)
    # publish_thread.daemon = True  # terminate when main thread exit
    # publish_thread.start()

class PathComplete(State):
    def __init__(self):
        State.__init__(self, outcomes=['success'])

    def execute(self, userdata):
        rospy.loginfo('###############################')
        rospy.loginfo('##### REACHED FINISH GATE #####')
        rospy.loginfo('###############################')
        return 'success'

class FollowWaypointsAction():
    _feedback = follow_waypoints.msg.WaypointsFeedback()
    _result   = follow_waypoints.msg.WaypointsResult()

    def __init__(self):
        self._action_name = rospy.get_param('~action_name','waypoints_action')
        self._as = actionlib.SimpleActionServer(self._action_name, follow_waypoints.msg.WaypointsAction, execute_cb=self.execute_cb)
        self._as.start()
        self._result_ready = False

    def execute_cb(self, goal):
        rospy.loginfo('Recieved path READY action goal')
        self._result_ready = False
        self.path_ready = True
        with open(output_file_path, 'w') as file:
            for current_pose in waypoints:
                file.write(str(current_pose.pose.pose.position.x) + ',' + str(current_pose.pose.pose.position.y) + ',' + str(current_pose.pose.pose.position.z) + ',' + str(current_pose.pose.pose.orientation.x) + ',' + str(current_pose.pose.pose.orientation.y) + ',' + str(current_pose.pose.pose.orientation.z) + ',' + str(current_pose.pose.pose.orientation.w)+ '\n')
        rospy.loginfo('poses written to '+ output_file_path)

        while not self._result_ready:
            continue
        if self._result.result == WaypointsResult.RESET:
            self._as.set_aborted(self._result)
        elif self._result.result == WaypointsResult.CANCELED:
            self._as.set_preempted(self._result)
        elif self._result.result == WaypointsResult.FAILED:
            self._as.set_aborted(self._result)
        elif self._result.result == WaypointsResult.TIMEOUT:
            self._as.set_aborted(self._result)
        elif self._result.result == WaypointsResult.SUCCEEDED:
            self._as.set_succeeded(self._result)

    def send_feedback(self, feedback):
        self._feedback.text = feedback
        self._as.publish_feedback(self._feedback)

    def set_result(self, result):
        self._result.result = result
        self._result_ready = True

def main():
    global fw
    rospy.init_node('follow_waypoints')
    fw = FollowWaypointsAction()
    sm = StateMachine(outcomes=['success'])

    with sm:
        StateMachine.add('GET_PATH', GetPath(),
                           transitions={'success':'FOLLOW_PATH'},
                           remapping={'waypoints':'waypoints'})
        StateMachine.add('FOLLOW_PATH', FollowPath(),
                           transitions={'success':'PATH_COMPLETE'},
                           remapping={'waypoints':'waypoints'})
        StateMachine.add('PATH_COMPLETE', PathComplete(),
                           transitions={'success':'GET_PATH'})

    outcome = sm.execute()
