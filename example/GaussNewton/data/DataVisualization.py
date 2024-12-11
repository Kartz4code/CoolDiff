#! /usr/bin/python3 

# @file example/GaussNewton/data/DataVisualization.py
#
# @copyright 2023-2024 Karthik Murali Madhavan Rathai
#
#
# This file is part of CoolDiff library.
#
# You can redistribute it and/or modify it under the terms of the GNU
# General Public License version 3 as published by the Free Software
# Foundation.
# 
# Licensees holding a valid commercial license may use this software
# in accordance with the commercial license agreement provided in
# conjunction with the software.  The terms and conditions of any such
# commercial license agreement shall govern, supersede, and render
# ineffective any application of the GPLv3 license to this software,
# notwithstanding of any reference thereto in the software or
# associated repository.

import matplotlib.pyplot as plt
from GaussNewtonData import *

in_file = g_input_data
out_file = g_output_data

x_in = []; y_in = []
x_out = []; y_out = []

with open(in_file, 'r') as file:
    for line in file:
        columns = line.split()
        if len(columns) == 2:
            x_in.append(float(columns[0]))
            y_in.append(float(columns[1]))

with open(out_file, 'r') as file:
    for line in file:
        columns = line.split()
        if len(columns) == 2:
            x_out.append(float(columns[0]))
            y_out.append(float(columns[1]))

plt.figure(figsize=(8, 6))
plt.scatter(x_in, y_in, marker='.', color='b', label='Input data')
plt.scatter(x_out, y_out, marker='.', color='r', label='Output data')

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('2D Plot of Data')
plt.legend()

plt.grid(True)
plt.show()
