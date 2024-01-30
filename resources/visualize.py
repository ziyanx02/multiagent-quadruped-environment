from urdfpy import URDF

urdf_path = "/home/ziyanx/python/multiagent-quadruped-environments/resources/objects/sheep.urdf"

robot = URDF.load(urdf_path)

robot.show()