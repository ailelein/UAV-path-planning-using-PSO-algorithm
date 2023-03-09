import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time
import copy
from sklearn.datasets._samples_generator import make_blobs
from itertools import product, combinations
import scipy.stats as st
import SDPC

X_sensor, y_true_sensor = make_blobs(n_samples=20000,centers=[[11, 11], [20, 20], [15, 38], [35, 15], [38, 35]],
                                     cluster_std=[4, 4, 5.1, 5, 5], random_state=0)
x_sensors = X_sensor[:, 0]
y_sensors = X_sensor[:, 1]
xx_sensors, yy_sensors = np.mgrid[0:50:50j, 0:50:50j]
positions_sensors = np.vstack([xx_sensors.ravel(), yy_sensors.ravel()])
values_sensor = np.vstack([x_sensors, y_sensors])
kernel_sensor = st.gaussian_kde(values_sensor)
f_sensor = np.reshape(kernel_sensor(positions_sensors).T, xx_sensors.shape)
zz_sensor = f_sensor * 22305.1 # 20 000 sensors
vector_sensor = np.vectorize(np.int32)
zz_sensor = vector_sensor(zz_sensor)

n_components = 3
X_surface, truth_surface = make_blobs(n_samples=300, centers=[[18, 18], [15, 35], [35, 30]], cluster_std=[6, 3, 4.5],
                                      random_state=42)
x_surface = X_surface[:, 0]
y_surface = X_surface[:, 1]
xx_surface, yy_surface = np.mgrid[-0.5:49.5:101j, -0.5:49.5:101j]
positions_surface = np.vstack([xx_surface.ravel(), yy_surface.ravel()])
values_surface = np.vstack([x_surface, y_surface])
kernel_surface = st.gaussian_kde(values_surface)
f_surface = np.reshape(kernel_surface(positions_surface).T, xx_surface.shape)
zz_surface = f_surface * 5000
z_fin_surface = []
a_surface = 1
while a_surface <= 99:
    z_surface = []
    for i in range(1, 100, 2):
        z_surface.append(zz_surface[a_surface][i])
    z_fin_surface.append(z_surface)
    a_surface += 2
z_axis_surface = np.array(z_fin_surface)

people_number = zz_sensor
z_mountain = z_axis_surface
NumUAV = 5
theta = math.pi / 3
noise = -80
moving_speed = 10
ValueMax = 2.0
ValueMin = 0.2
sensor_num_Max = 10
sensor_num_Min = 2
d_u_s_max = 265  # 1025
d_u_u_max = 1327  # change to 1327 or 839
w1, w2, w3 = 0.3, 0.35, 0.35
R = 3.5 # 120 secs to 2 mins
NOR_S = 124
NOR_T = 43.9
NOR_E = 75
em = 2
es = 1
ec = 0.03
a_coef = (ValueMax - ValueMin) / ((math.e ** R) - 1)
b_coef = ValueMin - a_coef

GBS_position = [0, 0, 0]
global_average_f = 0
global_average_ft = 0
global_average_fs = 0
global_average_fe = 0
global_time = 0

prohibited_zone = []
for a in range(40, 43):
    for b in range(40, 43):
        for c in range(10, 13):
            prohibited_zone.append([a, b, c])


def distance(x1, y1, z1, x2, y2, z2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) * 40
    return dist


def check_coverage(d_u_s_max, theta, x, y, z):
    covered_cells_1 = []
    covered_cells_2 = []
    covered_cells_3 = []
    covered_cells_final = []
    r_max = d_u_s_max * math.sin(theta / 2)
    for x_cell in range(0, 50):
        for y_cell in range(0, 50):
            r_k_i = math.sqrt((x - x_cell) ** 2 + (y - y_cell) ** 2) * 40
            if r_max >= r_k_i:
                covered_cells_1.append([x_cell, y_cell])
    for cell in covered_cells_1:
        d_k_i = distance(x, y, z, cell[0], cell[1], z_mountain[cell[0]][cell[1]])
        if (d_u_s_max >= d_k_i) and (z_mountain[cell[0]][cell[1]] < z):
            covered_cells_2.append([cell[0], cell[1]])
    for cell in covered_cells_2:
        d_k_i = distance(x, y, z, cell[0], cell[1], z_mountain[cell[0]][cell[1]])
        d = ((z - z_mountain[cell[0]][cell[1]]) * 40) / math.cos(theta / 2)
        if d_k_i <= d:
            covered_cells_3.append([cell[0], cell[1]])
    for cell in covered_cells_3:
        cell_j = SDPC.coordinate_and_compare_height(x, y, cell[0], cell[1])
        for cell_j_axis in cell_j:
            z_j_LOS_k_i = (z - z_mountain[cell[0]][cell[1]]) / (
                math.sqrt((x - cell[0]) ** 2 + (y - cell[1]) ** 2)) * math.sqrt(
                (cell_j_axis[0] - cell[0]) ** 2 + (cell_j_axis[1] - cell[1]) ** 2) + z_mountain[cell[0]][cell[1]]
            if z_mountain[cell_j_axis[0]][cell_j_axis[1]] < z_j_LOS_k_i:
                if [cell[0], cell[1]] not in covered_cells_final:
                    covered_cells_final.append([cell[0], cell[1]])
    return covered_cells_final


def SNR(x_uav, y_uav, z_uav, x_c_i, y_c_i):
    d_k_i = distance(x_uav, y_uav, z_uav, x_c_i, y_c_i, z_mountain[x_c_i][y_c_i])
    cal_snr = (10 * math.log10(10 / (9 * 16 * (math.pi ** 2) * (d_k_i ** 2)))) - noise
    return cal_snr


def sensing_function():
    for x_value in range(50):
        for y_value in range(50):
            if abs(time_of_each_UAV[current_uav] - prev_time[x_value][y_value]) <= R:
                v_value[x_value][y_value] = a_coef * math.exp(
                    time_of_each_UAV[current_uav] - prev_time[x_value][y_value]) + b_coef
            else:
                v_value[x_value][y_value] = ValueMax
    value_list.append(v_value[vsi_x][vsi_y])
    time_list.append(time_of_each_UAV[current_uav])


def number_of_sensor_function():
    for i in range(50):
        for j in range(50):
            if people_number[i][j] < 0:
                sensor_num[i][j] = 0
            elif 1 <= people_number[i][j] <= sensor_num_Max:
                sensor_num[i][j] = (((sensor_num_Max - sensor_num_Min) * people_number[i][j]) + (
                        sensor_num_Max * (sensor_num_Min - 1))) / (sensor_num_Max - 1)
            elif people_number[i][j] > sensor_num_Max:
                sensor_num[i][j] = sensor_num_Max


def F_S_function(Part_Position_x, Part_Position_y, Part_Position_z):
    all_covered_cells = []
    for all_uav_current_position in UAV_current_Position:
        each_uav_covered_cell = check_coverage(d_u_s_max, theta, all_uav_current_position[0],
                                               all_uav_current_position[1], all_uav_current_position[2])
        all_covered_cells.append(each_uav_covered_cell)
    PSO_particle_covered_cells = check_coverage(d_u_s_max, theta, Part_Position_x, Part_Position_y, Part_Position_z)
    particle_covered_cells = set()
    covered_cells = set()
    for i in PSO_particle_covered_cells:
        particle_covered_cells.add((i[0], i[1]))
    for uav_covered_cells in all_covered_cells:
        for j in uav_covered_cells:
            covered_cells.add((j[0], j[1]))
    not_included_in_other_uav = particle_covered_cells - covered_cells
    sum_F_S = 0
    for x_y_not_included_in_other_uav in not_included_in_other_uav:
        sum_F_S += v_value[x_y_not_included_in_other_uav[0]][x_y_not_included_in_other_uav[1]] * \
                   sensor_num[x_y_not_included_in_other_uav[0]][x_y_not_included_in_other_uav[1]]
    F_S = sum_F_S / NOR_S
    return F_S


def F_T_function(Part_Position_x, Part_Position_y, Part_Position_z, UAV_x, UAV_y, UAV_z):
    d_k_m = distance(UAV_x, UAV_y, UAV_z, Part_Position_x, Part_Position_y, Part_Position_z)
    T_moving = d_k_m / moving_speed
    F_T_cell = check_coverage(d_u_s_max, theta, Part_Position_x, Part_Position_y, Part_Position_z)
    T_stay = 0
    for cell in F_T_cell:
        snr1 = SNR(x_uav=UAV_x, y_uav=UAV_y, z_uav=UAV_z, x_c_i=cell[0], y_c_i=cell[1])
        p_g_c_i = 10 ** (-0.7 * math.e ** (0.05 * snr1 - 0.6))
        T_stay += people_number[cell[0]][cell[1]] * 0.02 * (1 / (1 - p_g_c_i))
    F_T = (T_moving + T_stay) / NOR_T
    return F_T


def F_E_function(Part_Position_x, Part_Position_y, Part_Position_z, UAV_x, UAV_y, UAV_z):
    d_k_m = distance(UAV_x, UAV_y, UAV_z, Part_Position_x, Part_Position_y, Part_Position_z)
    E_k_M = d_k_m / moving_speed * em
    E_k_S_cell = check_coverage(d_u_s_max, theta, Part_Position_x, Part_Position_y, Part_Position_z)
    E_k_S_cell_sum = 0
    E_k_C_sum = 0
    for cell in E_k_S_cell:
        snr1 = SNR(x_uav=UAV_x, y_uav=UAV_y, z_uav=UAV_z, x_c_i=cell[0], y_c_i=cell[1])
        p_g_c_i = 10 ** (-0.7 * math.e ** (0.05 * snr1 - 0.6))
        E_k_S_cell_sum += people_number[cell[0]][cell[1]] * 0.02 * (1 / (1 - p_g_c_i))
        E_k_C_sum += people_number[cell[0]][cell[1]] * (1 / (1 - p_g_c_i)) * ec
    E_k_S = E_k_S_cell_sum * es
    F_E = (E_k_M + E_k_S + E_k_C_sum) / NOR_E
    return F_E


def fitness_function(x, y, z, UAV_x, UAV_y, UAV_z):
    F_S = F_S_function(x, y, z)
    F_T = F_T_function(x, y, z, UAV_x, UAV_y, UAV_z)
    F_E = F_E_function(x, y, z, UAV_x, UAV_y, UAV_z)
    F = w1 * F_S - w2 * F_T - w3 * F_E
    return F


def check_sdpc_function(cell_j, x1, y1, z1, x2, y2, z2):
    check_sdpc = []
    for cell_j_axis in cell_j:
        z_j_LOS_uav_gbs = (z1 - z2) / (math.sqrt(
            (x1 - x2) ** 2 + (y1 - y2) ** 2)) * math.sqrt((cell_j_axis[0] - x2) ** 2 + (cell_j_axis[1] - y2) ** 2) + z2
        if z_mountain[cell_j_axis[0]][cell_j_axis[1]] < z_j_LOS_uav_gbs:
            check_sdpc.append(1)
        else:
            check_sdpc.append(0)
    return check_sdpc


simulation_time = 1
while simulation_time <2:
    print("simulation # ", simulation_time)
    start_time = time.time()
    v_value = np.ones((50, 50), float)
    v_value.fill(2.0)
    prev_time = np.zeros((50, 50), float)
    prev_time.fill(100)
    sensor_num = np.zeros((50, 50), int)
    number_of_sensor_function()
    UAV_current_Position = [[0, 0, 0] for _ in range(NumUAV)]
    time_of_each_UAV = [0, 0, 0, 0, 0]
    vsi_x = 0
    vsi_y = 0
    time_list = [0]
    value_list = [2]
    average_f_value = 0
    average_s_value = 0
    average_t_value = 0
    average_e_value = 0
    current_time_list= []
    current_time_fitness_value= []

    for current_uav in range(NumUAV):
        next_uav_position = []
        init_value_while_to_false = 0
        while init_value_while_to_false < 2:
            new_position_list = []
            new_position_list.append(random.randint(0, 49))
            new_position_list.append(random.randint(0, 49))
            new_position_list.append(random.uniform(0, 23.9))
            if (new_position_list not in prohibited_zone) and (new_position_list not in UAV_current_Position) and (
                    new_position_list[2] > z_mountain[new_position_list[0]][new_position_list[1]]):
                if current_uav == 0:
                    if (distance(new_position_list[0], new_position_list[1], new_position_list[2], GBS_position[0],
                                 GBS_position[1], GBS_position[2]) <= d_u_u_max) and (all(check_sdpc_function(
                            SDPC.coordinate_and_compare_height(new_position_list[0], new_position_list[1],
                                                               GBS_position[0], GBS_position[1]),
                            new_position_list[0], new_position_list[1], new_position_list[2], GBS_position[0],
                            GBS_position[1], GBS_position[2])) == 1):
                        next_uav_position = new_position_list
                        break
                    else:
                        continue
                else:
                    if (any(((distance(new_position_list[0], new_position_list[1], new_position_list[2],
                                       UAV_current_Position[num_gbs_uav][0], UAV_current_Position[num_gbs_uav][1],
                                       UAV_current_Position[num_gbs_uav][2])) <= 839) and (all(check_sdpc_function(
                        SDPC.coordinate_and_compare_height(new_position_list[0],
                                                           new_position_list[1],
                                                           UAV_current_Position[num_gbs_uav][0],
                                                           UAV_current_Position[num_gbs_uav][1]),
                        new_position_list[0], new_position_list[1], new_position_list[2],
                        UAV_current_Position[num_gbs_uav][0],
                        UAV_current_Position[num_gbs_uav][1],
                        UAV_current_Position[num_gbs_uav][2])) == 1) for num_gbs_uav in range(0, current_uav))) or (
                            distance(new_position_list[0], new_position_list[1], new_position_list[2],
                                     GBS_position[0], GBS_position[1], GBS_position[2]) <= d_u_u_max) and (
                            all(check_sdpc_function(
                                SDPC.coordinate_and_compare_height(new_position_list[0], new_position_list[1],
                                                                   GBS_position[0], GBS_position[1]),
                                new_position_list[0], new_position_list[1], new_position_list[2],
                                GBS_position[0], GBS_position[1],
                                GBS_position[2])) == 1):
                        next_uav_position = new_position_list
                        break
                    else:
                        continue

                # GBS_UAV_connection = []
                # for check_connection in range(5):
                #     if check_connection != current_uav:
                #         if distance(UAV_current_Position[check_connection][0],
                #                     UAV_current_Position[check_connection][1],
                #                     UAV_current_Position[check_connection][2], GBS_position[0], GBS_position[1],
                #                     GBS_position[2]) <= d_u_u_max:
                #             cell_j = SDPC.coordinate_and_compare_height(
                #                 UAV_current_Position[check_connection][0],
                #                 UAV_current_Position[check_connection][1], GBS_position[0], GBS_position[1])
                #             sdpc_list_check = check_sdpc_function(cell_j,
                #                                                   UAV_current_Position[check_connection][0],
                #                                                   UAV_current_Position[check_connection][1],
                #                                                   UAV_current_Position[check_connection][2],
                #                                                   GBS_position[0], GBS_position[1],
                #                                                   GBS_position[2])
                #             if all(sdpc_list_check) == 1:
                #                 GBS_UAV_connection.append(UAV_current_Position[check_connection])
                # if len(GBS_UAV_connection) >= 1:
                #     for con_num in GBS_UAV_connection:
                #         if distance(new_position_list[0], new_position_list[1], new_position_list[2],
                #                     con_num[0], con_num[1], con_num[2]) <= d_u_u_max:
                #             cell_j = SDPC.coordinate_and_compare_height(new_position_list[0], new_position_list[1],
                #                                                         con_num[0],
                #                                                         con_num[1])
                #             sdpc_list_check = check_sdpc_function(cell_j, new_position_list[0], new_position_list[1],
                #                                                   new_position_list[2],
                #                                                   con_num[0], con_num[1], con_num[2])
                #             if all(sdpc_list_check) == 1:
                #                 next_uav_position = new_position_list
                #                 init_value_while_to_false = 5
                #                 break
                # elif distance(new_position_list[0], new_position_list[1], new_position_list[2], GBS_position[0],
                #               GBS_position[1], GBS_position[2]) <= d_u_u_max:
                #     cell_j = SDPC.coordinate_and_compare_height(new_position_list[0], new_position_list[1],
                #                                                 GBS_position[0], GBS_position[1])
                #     sdpc_list_check = check_sdpc_function(cell_j, new_position_list[0], new_position_list[1],
                #                                           new_position_list[2],
                #                                           GBS_position[0], GBS_position[1], GBS_position[2])
                #     if all(sdpc_list_check) == 1:
                #         next_uav_position = new_position_list
                #         break
            #     else:
            #         continue
            # else:
            #     continue
        moving_time = ((math.sqrt((UAV_current_Position[current_uav][0] - next_uav_position[0]) ** 2 + (
                UAV_current_Position[current_uav][1] - next_uav_position[1]) ** 2 + (
                                          UAV_current_Position[current_uav][2] - next_uav_position[
                                      2]) ** 2)) * 40 / moving_speed) / 60
        check_coverage_next = check_coverage(d_u_s_max, theta, next_uav_position[0], next_uav_position[1],
                                             next_uav_position[2])
        if current_uav == 0:
            if (len(check_coverage_next) > 1) and (check_coverage_next[0] != [0, 0]):
                x_y_vsi = check_coverage_next[0]
                vsi_x = x_y_vsi[0]
                vsi_y = x_y_vsi[1]
        next_position_stay_time_sum = 0
        for cov_next in check_coverage_next:
            snr = SNR(next_uav_position[0], next_uav_position[1], next_uav_position[2], cov_next[0], cov_next[1])
            p_g_c_i = 10 ** (-0.7 * math.e ** (0.05 * snr - 0.6))
            next_stay_time_c_i = people_number[cov_next[0]][cov_next[1]] * 0.02 * 1 / (1 - p_g_c_i)
            next_position_stay_time_sum += next_stay_time_c_i
        next_position_stay_time = next_position_stay_time_sum / 60
        F_S = F_S_function(next_uav_position[0], next_uav_position[1], next_uav_position[2])
        F_T = F_T_function(next_uav_position[0], next_uav_position[1], next_uav_position[2],
                           UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                           UAV_current_Position[current_uav][2])
        F_E = F_E_function(next_uav_position[0], next_uav_position[1], next_uav_position[2],
                           UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                           UAV_current_Position[current_uav][2])
        F = fitness_function(next_uav_position[0], next_uav_position[1], next_uav_position[2],
                             UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                             UAV_current_Position[current_uav][2])

        time_of_each_UAV[current_uav] = time_of_each_UAV[current_uav] + moving_time + next_position_stay_time
        UAV_current_Position[current_uav] = next_uav_position
        # print(f"F_S, F_T, F_E of the next optimal position: {F_S, F_T, F_E}")
        # print(f"Total fitness F of the next optimal position: {F}")

    movement = 1
    while movement < 51:
        # print(f"Movement number: {movement}")
        # print(f"UAV_current_Position: {UAV_current_Position}")
        # if movement > 0:
        #     fig, ax = plt.subplots(nrows=1, ncols=1)
        #     ax.set_xlim(xmin=0, xmax=50)
        #     ax.set_ylim(ymin=0, ymax=50)
        #     ax.xaxis.set_major_locator(MultipleLocator(base=5))
        #     ax.yaxis.set_major_locator(MultipleLocator(base=5))
        #     plt.contour(xx_surface, yy_surface, zz_surface, 20, cmap='RdGy')
        #     plt.scatter(x_sensors, y_sensors, s=7, color='#495057', zorder=3)
        #     for uav_connection in range(5):
        #         if uav_connection == 0:
        #             color = 'blue'
        #         elif uav_connection == 1:
        #             color = 'green'
        #         elif uav_connection == 2:
        #             color = 'pink'
        #         elif uav_connection == 3:
        #             color = 'purple'
        #         elif uav_connection == 4:
        #             color = 'yellow'
        #         plt.scatter(UAV_current_Position[uav_connection][0], UAV_current_Position[uav_connection][1], s=100,
        #                     zorder=10, color=color)
        #         for con_num in range(5):
        #             if con_num != uav_connection:
        #                 if distance(UAV_current_Position[uav_connection][0], UAV_current_Position[uav_connection][1],
        #                             UAV_current_Position[uav_connection][2],
        #                             UAV_current_Position[con_num][0], UAV_current_Position[con_num][1],
        #                             UAV_current_Position[con_num][2]) <= d_u_u_max:
        #                     cell_j = SDPC.coordinate_and_compare_height(UAV_current_Position[uav_connection][0],
        #                                                                 UAV_current_Position[uav_connection][1],
        #                                                                 UAV_current_Position[con_num][0],
        #                                                                 UAV_current_Position[con_num][1])
        #                     sdpc_list_check = check_sdpc_function(cell_j, UAV_current_Position[uav_connection][0],
        #                                                           UAV_current_Position[uav_connection][1],
        #                                                           UAV_current_Position[uav_connection][2],
        #                                                           UAV_current_Position[con_num][0],
        #                                                           UAV_current_Position[con_num][1],
        #                                                           UAV_current_Position[con_num][2])
        #                     if all(sdpc_list_check) == 1:
        #                         plt.plot([UAV_current_Position[uav_connection][0], UAV_current_Position[con_num][0]],
        #                                  [UAV_current_Position[uav_connection][1], UAV_current_Position[con_num][1]],
        #                                  color='blue', zorder=8, marker='_', markersize=12)
        #         if distance(UAV_current_Position[uav_connection][0], UAV_current_Position[uav_connection][1],
        #                     UAV_current_Position[uav_connection][2],
        #                     GBS_position[0], GBS_position[1], GBS_position[2]) <= d_u_u_max:
        #             cell_j = SDPC.coordinate_and_compare_height(UAV_current_Position[uav_connection][0],
        #                                                         UAV_current_Position[uav_connection][1],
        #                                                         GBS_position[0], GBS_position[1])
        #             sdpc_list_check = check_sdpc_function(cell_j, UAV_current_Position[uav_connection][0],
        #                                                   UAV_current_Position[uav_connection][1],
        #                                                   UAV_current_Position[uav_connection][2],
        #                                                   GBS_position[0], GBS_position[1], GBS_position[2])
        #             if all(sdpc_list_check) == 1:
        #                 plt.plot([UAV_current_Position[uav_connection][0], 0],
        #                          [UAV_current_Position[uav_connection][1], 0],
        #                          color='blue', zorder=8, marker='_', markersize=12)
        #

        min_time = min(every_time for every_time in time_of_each_UAV)
        current_uav = time_of_each_UAV.index(min_time)
        check_coverage_current_uav = check_coverage(d_u_s_max, theta, UAV_current_Position[current_uav][0],
                                                    UAV_current_Position[current_uav][1],
                                                    UAV_current_Position[current_uav][2])
        sensing_function()
        for cov in check_coverage_current_uav:
            prev_time[cov[0]][cov[1]] = time_of_each_UAV[current_uav]
        sensing_function()


        next_uav_position = []
        init_value_while_to_false = 0
        while init_value_while_to_false < 2:
            new_position_list = []
            new_position_list.append(random.randint(0, 49))
            new_position_list.append(random.randint(0, 49))
            new_position_list.append(random.uniform(0, 23.9))
            if (new_position_list not in prohibited_zone) and (new_position_list not in UAV_current_Position) and (
                    new_position_list[2] > z_mountain[new_position_list[0]][new_position_list[1]]):
                GBS_UAV_connection = []
                for num_gbs_uav in range(5):
                    if num_gbs_uav != current_uav:
                        if (distance(UAV_current_Position[num_gbs_uav][0],
                                     UAV_current_Position[num_gbs_uav][1],
                                     UAV_current_Position[num_gbs_uav][2], GBS_position[0], GBS_position[1],
                                     GBS_position[2]) <= d_u_u_max) and (all(check_sdpc_function(
                            SDPC.coordinate_and_compare_height(UAV_current_Position[num_gbs_uav][0],
                                                               UAV_current_Position[num_gbs_uav][1],
                                                               GBS_position[0], GBS_position[1]),
                            UAV_current_Position[num_gbs_uav][0],
                            UAV_current_Position[num_gbs_uav][1],
                            UAV_current_Position[num_gbs_uav][2],
                            GBS_position[0], GBS_position[1], GBS_position[2])) == 1):
                            GBS_UAV_connection.append(num_gbs_uav)
                if len(GBS_UAV_connection) >= 1:
                    exitFlag = False
                    for connect_to_pso in GBS_UAV_connection:
                        if (distance(UAV_current_Position[connect_to_pso][0],
                                     UAV_current_Position[connect_to_pso][1],
                                     UAV_current_Position[connect_to_pso][2], new_position_list[0],
                                     new_position_list[1], new_position_list[2]) <= d_u_u_max) and (
                                all(check_sdpc_function(
                                    SDPC.coordinate_and_compare_height(UAV_current_Position[connect_to_pso][0],
                                                                       UAV_current_Position[connect_to_pso][1],
                                                                       new_position_list[0],
                                                                       new_position_list[1]),
                                    UAV_current_Position[connect_to_pso][0],
                                    UAV_current_Position[connect_to_pso][1],
                                    UAV_current_Position[connect_to_pso][2],
                                    new_position_list[0], new_position_list[1],
                                    new_position_list[2])) == 1):

                            next_uav_position = new_position_list
                            init_value_while_to_false = 5
                            break
                        else:
                            exitFlag2 = False
                            for connect_to_pso2 in range(5):
                                if (connect_to_pso2 != connect_to_pso) and (connect_to_pso2 != current_uav):
                                    if (distance(UAV_current_Position[connect_to_pso][0],
                                                 UAV_current_Position[connect_to_pso][1],
                                                 UAV_current_Position[connect_to_pso][2],
                                                 UAV_current_Position[connect_to_pso2][0],
                                                 UAV_current_Position[connect_to_pso2][1],
                                                 UAV_current_Position[connect_to_pso2][2]) <= d_u_u_max) and (
                                            all(check_sdpc_function(
                                                SDPC.coordinate_and_compare_height(
                                                    UAV_current_Position[connect_to_pso][0],
                                                    UAV_current_Position[connect_to_pso][1],
                                                    UAV_current_Position[connect_to_pso2][0],
                                                    UAV_current_Position[connect_to_pso2][
                                                        1]),
                                                UAV_current_Position[connect_to_pso][0],
                                                UAV_current_Position[connect_to_pso][1],
                                                UAV_current_Position[connect_to_pso][2],
                                                UAV_current_Position[connect_to_pso2][0],
                                                UAV_current_Position[connect_to_pso2][1],
                                                UAV_current_Position[connect_to_pso2][2])) == 1):

                                        if (distance(UAV_current_Position[connect_to_pso2][0],
                                                     UAV_current_Position[connect_to_pso2][1],
                                                     UAV_current_Position[connect_to_pso2][2],
                                                     new_position_list[0],
                                                     new_position_list[1],
                                                     new_position_list[2]) <= d_u_u_max) and (
                                                all(check_sdpc_function(
                                                    SDPC.coordinate_and_compare_height(
                                                        UAV_current_Position[connect_to_pso2][0],
                                                        UAV_current_Position[connect_to_pso2][1],
                                                        new_position_list[0], new_position_list[1]),
                                                    UAV_current_Position[connect_to_pso2][0],
                                                    UAV_current_Position[connect_to_pso2][1],
                                                    UAV_current_Position[connect_to_pso2][2],
                                                    new_position_list[0], new_position_list[1],
                                                    new_position_list[2])) == 1):

                                            next_uav_position = new_position_list
                                            init_value_while_to_false = 5
                                            exitFlag = True
                                            break
                                        else:
                                            exitFlag3 = False
                                            for connect_to_pso3 in range(5):
                                                if (connect_to_pso3 != current_uav) and (
                                                        connect_to_pso3 != connect_to_pso2) and (
                                                        connect_to_pso3 != connect_to_pso):
                                                    if (distance(UAV_current_Position[connect_to_pso3][0],
                                                                 UAV_current_Position[connect_to_pso3][1],
                                                                 UAV_current_Position[connect_to_pso3][2],
                                                                 UAV_current_Position[connect_to_pso2][0],
                                                                 UAV_current_Position[connect_to_pso2][1],
                                                                 UAV_current_Position[connect_to_pso2][
                                                                     2]) <= d_u_u_max) and (
                                                            all(check_sdpc_function(
                                                                SDPC.coordinate_and_compare_height(
                                                                    UAV_current_Position[connect_to_pso3][0],
                                                                    UAV_current_Position[connect_to_pso3][1],
                                                                    UAV_current_Position[connect_to_pso2][0],
                                                                    UAV_current_Position[connect_to_pso2][
                                                                        1]),
                                                                UAV_current_Position[connect_to_pso3][0],
                                                                UAV_current_Position[connect_to_pso3][1],
                                                                UAV_current_Position[connect_to_pso3][2],
                                                                UAV_current_Position[connect_to_pso2][0],
                                                                UAV_current_Position[connect_to_pso2][1],
                                                                UAV_current_Position[connect_to_pso2][2])) == 1):
                                                        if (distance(UAV_current_Position[connect_to_pso3][0],
                                                                     UAV_current_Position[connect_to_pso3][1],
                                                                     UAV_current_Position[connect_to_pso3][2],
                                                                     new_position_list[0],
                                                                     new_position_list[1],
                                                                     new_position_list[2]) <= d_u_u_max) and (
                                                                all(check_sdpc_function(
                                                                    SDPC.coordinate_and_compare_height(
                                                                        UAV_current_Position[connect_to_pso3][0],
                                                                        UAV_current_Position[connect_to_pso3][1],
                                                                        new_position_list[0],
                                                                        new_position_list[1]),
                                                                    UAV_current_Position[connect_to_pso3][0],
                                                                    UAV_current_Position[connect_to_pso3][1],
                                                                    UAV_current_Position[connect_to_pso3][2],
                                                                    new_position_list[0],
                                                                    new_position_list[1],
                                                                    new_position_list[2])) == 1):
                                                            next_uav_position = new_position_list
                                                            init_value_while_to_false = 5
                                                            exitFlag = True
                                                            exitFlag2 = True
                                                            break
                                                        else:
                                                            for connect_to_pso4 in range(5):
                                                                if (connect_to_pso4 != current_uav) and (
                                                                        connect_to_pso4 != connect_to_pso2) and (
                                                                        connect_to_pso4 != connect_to_pso) and (
                                                                        connect_to_pso4 != connect_to_pso3):
                                                                    if (distance(
                                                                            UAV_current_Position[connect_to_pso3][
                                                                                0],
                                                                            UAV_current_Position[connect_to_pso3][
                                                                                1],
                                                                            UAV_current_Position[connect_to_pso3][
                                                                                2],
                                                                            UAV_current_Position[connect_to_pso2][
                                                                                0],
                                                                            UAV_current_Position[connect_to_pso2][
                                                                                1],
                                                                            UAV_current_Position[connect_to_pso2][
                                                                                2]) <= d_u_u_max) and (
                                                                            all(check_sdpc_function(
                                                                                SDPC.coordinate_and_compare_height(
                                                                                    UAV_current_Position[
                                                                                        connect_to_pso3][
                                                                                        0],
                                                                                    UAV_current_Position[
                                                                                        connect_to_pso3][
                                                                                        1],
                                                                                    UAV_current_Position[
                                                                                        connect_to_pso4][
                                                                                        0],
                                                                                    UAV_current_Position[
                                                                                        connect_to_pso4][
                                                                                        1]),
                                                                                UAV_current_Position[
                                                                                    connect_to_pso3][
                                                                                    0],
                                                                                UAV_current_Position[
                                                                                    connect_to_pso3][
                                                                                    1],
                                                                                UAV_current_Position[
                                                                                    connect_to_pso3][
                                                                                    2],
                                                                                UAV_current_Position[
                                                                                    connect_to_pso4][
                                                                                    0],
                                                                                UAV_current_Position[
                                                                                    connect_to_pso4][
                                                                                    1],
                                                                                UAV_current_Position[
                                                                                    connect_to_pso4][
                                                                                    2])) == 1):
                                                                        if (distance(
                                                                                UAV_current_Position[
                                                                                    connect_to_pso4][
                                                                                    0],
                                                                                UAV_current_Position[
                                                                                    connect_to_pso4][
                                                                                    1],
                                                                                UAV_current_Position[
                                                                                    connect_to_pso4][
                                                                                    2],
                                                                                new_position_list[0],
                                                                                new_position_list[1],
                                                                                new_position_list[
                                                                                    2]) <= d_u_u_max) and (
                                                                                all(check_sdpc_function(
                                                                                    SDPC.coordinate_and_compare_height(
                                                                                        UAV_current_Position[
                                                                                            connect_to_pso4][0],
                                                                                        UAV_current_Position[
                                                                                            connect_to_pso4][1],
                                                                                        new_position_list[0],
                                                                                        new_position_list[1]),
                                                                                    UAV_current_Position[
                                                                                        connect_to_pso4][
                                                                                        0],
                                                                                    UAV_current_Position[
                                                                                        connect_to_pso4][
                                                                                        1],
                                                                                    UAV_current_Position[
                                                                                        connect_to_pso4][
                                                                                        2],
                                                                                    new_position_list[0],
                                                                                    new_position_list[1],
                                                                                    new_position_list[2])) == 1):
                                                                            next_uav_position = new_position_list
                                                                            init_value_while_to_false = 5
                                                                            exitFlag = True
                                                                            exitFlag2 = True
                                                                            exitFlag3 = True
                                                                            break
                                                if (exitFlag3):
                                                    break
                                if (exitFlag2):
                                    break
                        if (exitFlag):
                            break
                else:
                    if (distance(new_position_list[0],
                                 new_position_list[1],
                                 new_position_list[2], GBS_position[0], GBS_position[1],
                                 GBS_position[2]) <= d_u_u_max) and (all(check_sdpc_function(
                        SDPC.coordinate_and_compare_height(new_position_list[0],
                                                           new_position_list[1],
                                                           GBS_position[0], GBS_position[1]),
                        new_position_list[0],new_position_list[1], new_position_list[2], GBS_position[0], GBS_position[1],
                        GBS_position[2])) == 1):
                        old_uav_to_other_uav_con = []
                        for second_check in range(5):
                            if second_check != current_uav:
                                if (distance(UAV_current_Position[second_check][0],
                                             UAV_current_Position[second_check][1],
                                             UAV_current_Position[second_check][2],
                                             UAV_current_Position[current_uav][0],
                                             UAV_current_Position[current_uav][1],
                                             UAV_current_Position[current_uav][2]) <= d_u_u_max) and (
                                        all(check_sdpc_function(
                                            SDPC.coordinate_and_compare_height(
                                                UAV_current_Position[second_check][0],
                                                UAV_current_Position[second_check][1],
                                                UAV_current_Position[current_uav][0],
                                                UAV_current_Position[current_uav][1]),
                                            UAV_current_Position[second_check][0],
                                            UAV_current_Position[second_check][1],
                                            UAV_current_Position[second_check][2],
                                            UAV_current_Position[current_uav][0],
                                            UAV_current_Position[current_uav][1],
                                            UAV_current_Position[current_uav][2])) == 1):
                                    old_uav_to_other_uav_con.append(second_check)

                        #
                        # only_to_current_uav_connect = []
                        # for con_elem in old_uav_to_other_uav_con:
                        #     for uav_num_without_some in range(5):
                        #         if uav_num_without_some != current_uav and uav_num_without_some != con_elem

                        # for con_elem in old_uav_to_other_uav_con:
                        if (all((distance(UAV_current_Position[con_elem][0],
                                     UAV_current_Position[con_elem][1],
                                     UAV_current_Position[con_elem][2], new_position_list[0],
                                     new_position_list[1],new_position_list[2])) <= d_u_u_max for con_elem in old_uav_to_other_uav_con)) and (all
                            (all(check_sdpc_function(
                                    SDPC.coordinate_and_compare_height(
                                        UAV_current_Position[con_elem][0],
                                        UAV_current_Position[con_elem][1],
                                        new_position_list[0], new_position_list[1]),
                                    UAV_current_Position[con_elem][0],
                                    UAV_current_Position[con_elem][1],
                                    UAV_current_Position[con_elem][2],
                                    new_position_list[0], new_position_list[1],
                                    new_position_list[2])) == 1 for con_elem in old_uav_to_other_uav_con)):
                            next_uav_position = new_position_list
                            init_value_while_to_false = 5
                            break



            #     GBS_UAV_connection = []
            #     for check_connection in range(5):
            #         if check_connection != current_uav:
            #             if distance(UAV_current_Position[check_connection][0],
            #                         UAV_current_Position[check_connection][1],
            #                         UAV_current_Position[check_connection][2], GBS_position[0], GBS_position[1],
            #                         GBS_position[2]) <= d_u_u_max:
            #                 cell_j = SDPC.coordinate_and_compare_height(
            #                     UAV_current_Position[check_connection][0],
            #                     UAV_current_Position[check_connection][1], GBS_position[0], GBS_position[1])
            #                 sdpc_list_check = check_sdpc_function(cell_j,
            #                                                       UAV_current_Position[check_connection][0],
            #                                                       UAV_current_Position[check_connection][1],
            #                                                       UAV_current_Position[check_connection][2],
            #                                                       GBS_position[0], GBS_position[1],
            #                                                       GBS_position[2])
            #                 if all(sdpc_list_check) == 1:
            #                     GBS_UAV_connection.append(UAV_current_Position[check_connection])
            #     if len(GBS_UAV_connection) >= 1:
            #         for con_num in GBS_UAV_connection:
            #             if distance(new_position_list[0], new_position_list[1], new_position_list[2],
            #                         con_num[0], con_num[1], con_num[2]) <= d_u_u_max:
            #                 cell_j = SDPC.coordinate_and_compare_height(new_position_list[0], new_position_list[1],
            #                                                             con_num[0],
            #                                                             con_num[1])
            #                 sdpc_list_check = check_sdpc_function(cell_j, new_position_list[0], new_position_list[1],
            #                                                       new_position_list[2],
            #                                                       con_num[0], con_num[1], con_num[2])
            #                 if all(sdpc_list_check) == 1:
            #                     next_uav_position = new_position_list
            #                     init_value_while_to_false = 5
            #                     break
            #     elif distance(new_position_list[0], new_position_list[1], new_position_list[2], GBS_position[0],
            #                   GBS_position[1], GBS_position[2]) <= d_u_u_max:
            #         cell_j = SDPC.coordinate_and_compare_height(new_position_list[0], new_position_list[1],
            #                                                     GBS_position[0], GBS_position[1])
            #         sdpc_list_check = check_sdpc_function(cell_j, new_position_list[0], new_position_list[1],
            #                                               new_position_list[2],
            #                                               GBS_position[0], GBS_position[1], GBS_position[2])
            #         if all(sdpc_list_check) == 1:
            #             next_uav_position = new_position_list
            #             break
            #     else:
            #         continue
            # else:
            #     continue
        moving_time = ((math.sqrt((UAV_current_Position[current_uav][0] - next_uav_position[0]) ** 2 + (
                UAV_current_Position[current_uav][1] - next_uav_position[1]) ** 2 + (
                                          UAV_current_Position[current_uav][2] - next_uav_position[
                                      2]) ** 2)) * 40 / moving_speed) / 60
        check_coverage_next = check_coverage(d_u_s_max, theta, next_uav_position[0], next_uav_position[1],
                                             next_uav_position[2])

        next_position_stay_time_sum = 0
        for cov_next in check_coverage_next:
            snr = SNR(next_uav_position[0], next_uav_position[1], next_uav_position[2], cov_next[0], cov_next[1])
            p_g_c_i = 10 ** (-0.7 * math.e ** (0.05 * snr - 0.6))
            next_stay_time_c_i = people_number[cov_next[0]][cov_next[1]] * 0.02 * 1 / (1 - p_g_c_i)
            next_position_stay_time_sum += next_stay_time_c_i
        next_position_stay_time = next_position_stay_time_sum / 60
        F_S = F_S_function(next_uav_position[0], next_uav_position[1], next_uav_position[2])
        average_s_value += F_S
        F_T = F_T_function(next_uav_position[0], next_uav_position[1], next_uav_position[2],
                           UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                           UAV_current_Position[current_uav][2])
        average_t_value += F_T
        F_E = F_E_function(next_uav_position[0], next_uav_position[1], next_uav_position[2],
                           UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                           UAV_current_Position[current_uav][2])
        average_e_value += F_E
        F = fitness_function(next_uav_position[0], next_uav_position[1], next_uav_position[2],
                             UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                             UAV_current_Position[current_uav][2])
        average_f_value += F
        time_of_each_UAV[current_uav] = time_of_each_UAV[current_uav] + moving_time + next_position_stay_time
        UAV_current_Position[current_uav] = next_uav_position
        # print(f"F_S, F_T, F_E of the next optimal position: {F_S, F_T, F_E}")
        # print(f"Total fitness F of the next optimal position: {F}")
        current_time_list.append(time_of_each_UAV[current_uav])
        current_time_fitness_value.append(F)
        movement += 1
    print("final average of 50 movement times")
    print(f"average fitness value of 50 movement time: {average_f_value / 50}")
    print(f"average sensing value {average_s_value / 50}")
    print(f"average time value {average_t_value / 50}")
    print(f"average energy value {average_e_value / 50}")
    end_time = time.time()
    run_time_program = end_time - start_time
    print(f'Run time of the program is {run_time_program}')
    print(current_time_list, "time")
    print( current_time_fitness_value, "value")
    global_average_f += average_f_value / 50
    global_average_fs += average_s_value / 50
    global_average_fe += average_e_value / 50
    global_average_ft += average_t_value / 50
    global_time += run_time_program
    simulation_time += 1
# plt.show()
print(f"Average optimal fitness function after 5 simulations: {global_average_f / 5}")
print(f"Average sensing fitness function after 5 simulations: {global_average_fs / 5}")
print(f"Average time fitness function after 5 simulations: {global_average_ft / 5}")
print(f"Average energy fitness function after 5 simulations: {global_average_fe / 5}")
print(f"Average computation time: {global_time / 5}")

