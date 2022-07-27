from utils.general_util import *
import numpy as np

view_matrix = get_view_matrix([0, 1, 3],[0, 0, 0],[0, 1 , 0])

print(np.array(view_matrix).reshape(4,4).T)


view_matrix = np.array(view_matrix).reshape(4,4).T

print(np.linalg.inv(view_matrix[:3,:3]))

print(np.linalg.inv(view_matrix))

