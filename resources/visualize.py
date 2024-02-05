from urdfpy import URDF

urdf_path = "/home/ziyanx/python/multiagent-quadruped-environments/resources/objects/sheep.urdf"
urdf_path = "/home/ziyanx/python/multiagent-quadruped-environments/resources/robots/go1/urdf/go1.urdf"

robot = URDF.load(urdf_path)

robot.show()