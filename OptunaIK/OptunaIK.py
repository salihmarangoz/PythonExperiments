#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import  Header
from moveit_msgs.srv import GetPositionFK
import optuna
from visualization_msgs.msg import InteractiveMarkerUpdate

MOVEGROUP_NAME = "xarm7"
joint_max_position = [6.283185,2.094400,6.283185,3.927000,6.283185,3.141593,6.283185]
joint_min_position = [-6.283185,-2.059000,-6.283185,-0.191980,-6.283185,-1.692970,-6.283185]
find_local_solution = True

def compute_fk(header, fk_link_names, robot_state):
    try:
        f = rospy.ServiceProxy('compute_fk', GetPositionFK)
        result = f(header, fk_link_names, robot_state)
        if result.error_code.val != 1:
          return None
        return result
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def toDisplayRobotState(robot_state):
  drs = moveit_msgs.msg.DisplayRobotState()
  drs.state = robot_state
  return drs

class OptunaInverseKinematics(object):
  def __init__(self):
    super(OptunaInverseKinematics, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('optuna_invere_kinematics', anonymous=True)

    # init moveit
    rospy.wait_for_service('compute_fk')
    self.robot = moveit_commander.RobotCommander()
    self.move_group = moveit_commander.MoveGroupCommander(MOVEGROUP_NAME)
    self.robot_state_publisher = rospy.Publisher('/move_group/demo_robot_state',
                                                   moveit_msgs.msg.DisplayRobotState,
                                                   queue_size=20)

    self.previous_trials = []
    self.target_x = 0.0
    self.target_y = 0.0
    self.target_z = 0.0
    self.is_moved = False

    # Misc variables
    self.planning_frame = self.move_group.get_planning_frame()
    self.eef_link = self.move_group.get_end_effector_link()
    self.group_names = self.robot.get_group_names()
    self.previous_state = self.robot.get_current_state()

    self.endpoint_target_subscriber = rospy.Subscriber("/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update", InteractiveMarkerUpdate, self.endpoint_target_callback)

  def endpoint_target_callback(self, msg):
    if (len(msg.poses) > 0):
      if len(self.previous_trials):
        if self.target_x != msg.poses[0].pose.position.x or \
           self.target_y != msg.poses[0].pose.position.y or \
           self.target_z != msg.poses[0].pose.position.z:
            self.is_moved = True
      self.target_x = msg.poses[0].pose.position.x
      self.target_y = msg.poses[0].pose.position.y
      self.target_z = msg.poses[0].pose.position.z

  def demo(self):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    while not rospy.is_shutdown():
      target_x = self.target_x
      target_y = self.target_y
      target_z = self.target_z
      if self.is_moved:
        self.is_moved = False
        self.previous_trials = []
      is_moved = self.is_moved

      study = optuna.create_study(sampler=optuna.samplers.TPESampler())
      if len(self.previous_trials)>0 and not is_moved:
        study.add_trials(self.previous_trials)

      # optimize
      func = lambda trial: self.objective(trial, target_x, target_y, target_z)
      if find_local_solution:
        study.optimize(func, timeout=0.01, n_jobs=32)
      else:
        study.optimize(func, timeout=0.1, n_jobs=32)

      best_params = study.best_params
      self.previous_trials = study.best_trials

      print("previous_trials:", len(self.previous_trials))

      # publish state for rviz
      self.previous_state.joint_state.position = tuple(best_params.values())
      self.robot_state_publisher.publish(toDisplayRobotState(self.previous_state))

    rospy.spin()

  def objective(self, trial, target_x, target_y, target_z):
    robot_state = copy.deepcopy(self.previous_state)

    # sample joint values
    new_joint_values = []
    for i, j in enumerate(self.previous_state.joint_state.name):
      min_val = self.previous_state.joint_state.position[i] - 0.15
      if joint_min_position[i] > min_val:
        min_val = joint_min_position[i]
      max_val = self.previous_state.joint_state.position[i] + 0.15
      if joint_max_position[i] < max_val:
        max_val = joint_max_position[i]

      if find_local_solution:
        new_joint_values.append( trial.suggest_float(j, min_val, max_val) )
      else:
        new_joint_values.append( trial.suggest_float(j, joint_min_position[i], joint_max_position[i]) )

    # compute forward kinematics with the new joint values
    robot_state.joint_state.position = tuple(new_joint_values)
    header = Header()
    header.frame_id = self.planning_frame
    fk_link_names = [self.eef_link]
    endpoint = compute_fk(header, fk_link_names, robot_state)

    return (endpoint.pose_stamped[0].pose.position.x - target_x)**2 + \
           (endpoint.pose_stamped[0].pose.position.y - target_y)**2 + \
           (endpoint.pose_stamped[0].pose.position.z - target_z)**2

def main():
  tutorial = OptunaInverseKinematics()
  tutorial.demo()

if __name__ == '__main__':
  main()
