prompt_configs = {
  "PnP": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                          When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                          The very beginning steps of this task will be approaching robot arms to the object, and then it will grasp the object. After that, the robot will put the object into the target place while holding the object. But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                          The last steps of this task will be retreating robot arm from objects. Answer the question as the following format, which is a python dictonary:
                          {{"State": <your answer>,
                           "Next Action": <your answer>,
                          "Potential Error": <your answer>,
                          "Error Avoidance": <your answer> }}""",
  "turnONandOff": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                        When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                        The very beginning steps of this task will be approaching robot arms to the object, and then it will grasp the knob if it needs to turn on or off a stove, or it will move facuet if it needs to turn on or off a facuet. The robot will grasp, rorate, touch or move the facuet or knob until the task finished. But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                        Answer the question as the following format, which is a python dictonary:
                        {{"State": <your answer>,
                         "Next Action": <your answer>,
                        "Potential Error": <your answer>,
                        "Error Avoidance": <your answer> }}""",
  "turn_spout": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                      When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                      The very beginning steps of this task will be approaching robot arms to the spout, and then it will move the spout to the designated direction. But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                      Answer the question as the following format, which is a python dictonary:
                      {{"State": <your answer>,
                       "Next Action": <your answer>,
                      "Potential Error": <your answer>,
                      "Error Avoidance": <your answer> }}""",
  "pressCoffeeButton": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                     When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                     The very beginning steps of this task will be approaching robot arms to the coffee maker's button, and then it will press the button, But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                     Answer the question as the following format, which is a python dictonary:
                     {{"State": <your answer>,
                      "Next Action": <your answer>,
                     "Potential Error": <your answer>,
                     "Error Avoidance": <your answer> }}""",
  "pressMicronwaveButton":"""these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                     When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                     The very beginning steps of this task will be approaching robot arms to microwave's buttion, and then it will press the button to turn on or turn off by the instruction. But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                     Answer the question as the following format, which is a python dictonary:
                     {{"State": <your answer>,
                      "Next Action": <your answer>,
                     "Potential Error": <your answer>,
                     "Error Avoidance": <your answer> }}""",
  "door_and_drawer": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                       When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                       The very beginning steps of this task will be approaching robot gripper to the handle, and then it will grasp the handle and move it until reach the target position. If it need to close or open a double door, it need to move the gripper to another door after finishing the first one. But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                       Answer the question as the following format, which is a python dictonary:
                       {{"State": <your answer>,
                        "Next Action": <your answer>,
                       "Potential Error": <your answer>,
                       "Error Avoidance": <your answer> }}""",
  "PnP_release_no_retreat": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                        When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                        The very beginning steps of this task will be approaching robot arms to the object, and then it will grasp the object. After that, the robot will hold and put the object into the target place. After putting the object in the right place, the robot needs to release the gripper. But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                        Answer the question as the following format, which is a python dictonary:
                        {{"State": <your answer>,
                         "Next Action": <your answer>,
                        "Potential Error": <your answer>,
                        "Error Avoidance": <your answer> }}""",
  "without_complete_rate": """these three images are left, mid, and right images of a robot who is executing the task {task}.Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                          Answer the question as the following format:
                          State:
                          Next Action:
                          Potential Error:
                          Error Avoidance:""",
  "only_action": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what next action the robot here shuld take. Make sure the answer is clear, specific and less than 20 words. Don't repeat any information that is already given in the question.
                          When answer the question, consider the task complete rate, the number is 0 to 100 and it indicates what phase the robot is current in.
                          The very beginning steps of this task will be approaching robot arms to the object, and then it will grasp the object. After that, the robot will put the object into the target place while holding the object. But be careful, the complete rate is not 100% accurate, so the robot may not be in the exact state as the complete rate indicates. 
                          The last steps of this task will be retreating robot arm from objects. Answer the question by giving the next action the robot should take directly no any other information."""
}

