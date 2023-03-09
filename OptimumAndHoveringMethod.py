import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import time
from sklearn.datasets._samples_generator import make_blobs
import scipy.stats as st
import SDPC
n_components = 3
X_surface, truth_surface = make_blobs(n_samples=300, centers=[[0.72, 0.72], [0.6, 1.4], [1.4, 1.20]], cluster_std=[0.24, 0.12, 0.18], random_state=42)
x_surface = X_surface[:, 0]
y_surface = X_surface[:, 1]
xx_surface, yy_surface = np.mgrid[-0.02:1.98:51j, -0.02:1.98:51j]
positions_surface = np.vstack([xx_surface.ravel(), yy_surface.ravel()])
values_surface = np.vstack([x_surface, y_surface])
kernel_surface = st.gaussian_kde(values_surface)
f_surface = np.reshape(kernel_surface(positions_surface).T, xx_surface.shape)
zz_surface = f_surface * 0.35

X_sensor, y_true_sensor = make_blobs(n_samples=20000,centers=[[0.1, 0.1], [0.5, 0.5], [0.6, 1.25], [1.3, 0.6], [1.52, 1.4]],
                                     cluster_std=[0.2, 0.16, 0.204, 0.18, 0.2], random_state=0)
x_sensors = X_sensor[:, 0]
y_sensors = X_sensor[:, 1]
xx_sensors, yy_sensors = np.mgrid[0:1.96:50j, 0:1.96:50j]
positions_sensors = np.vstack([xx_sensors.ravel(), yy_sensors.ravel()])
values_sensor = np.vstack([x_sensors, y_sensors])
kernel_sensor = st.gaussian_kde(values_sensor)
f_sensor = np.reshape(kernel_sensor(positions_sensors).T, xx_sensors.shape)
zz_sensor = f_sensor * 37.865 # 20 000 sensors

# zz_sensor = f_sensor * 19.919 # 10 000 sensors
# zz_sensor = f_sensor * 10.859# 5 000 sensors

vector_sensor = np.vectorize(np.int32)
zz_sensor = vector_sensor(zz_sensor)

people_number = zz_sensor
z_mountain = np.delete(zz_surface, 0, 0) # delete the first line of  -0.02. It was added in order to male '0' as a center of the cell



prohibited_zone = []
for a in range(35, 38):
    for b in range(35, 38):
        for c in np.arange(0.7, 0.8):
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
        if d_u_s_max >= d_k_i:
            covered_cells_2.append([cell[0], cell[1]])
    for cell in covered_cells_2:
        d_k_i = distance(x, y, z, cell[0], cell[1], z_mountain[cell[0]][cell[1]])
        d = ((z - z_mountain[cell[0]][cell[1]]) * 40) / math.cos(theta / 2)
        if (d_k_i <= d) and (z_mountain[cell[0]][cell[1]] <= z):
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


def sensing_function(Part_Position_x, Part_Position_y, Part_Position_z, UAV_x, UAV_y, UAV_z):
    d_k_m = distance(UAV_x, UAV_y, UAV_z, Part_Position_x, Part_Position_y, Part_Position_z)
    T_moving = d_k_m / moving_speed / 60
    v_value_copy = copy.deepcopy(v_value)
    for x_value in range(50):
        for y_value in range(50):
            if abs((time_of_each_UAV[current_uav] + T_moving) - prev_time[x_value][y_value]) <= R:
                v_value_copy[x_value][y_value] = a_coef * math.exp(
                    (time_of_each_UAV[current_uav] + T_moving) - prev_time[x_value][y_value]) + b_coef
            else:
                v_value_copy[x_value][y_value] = ValueMax
    return v_value_copy


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


def F_S_function(Part_Position_x, Part_Position_y, Part_Position_z, UAV_x, UAV_y, UAV_z):
    copy_value = sensing_function(Part_Position_x, Part_Position_y, Part_Position_z, UAV_x, UAV_y, UAV_z)
    all_covered_cells = []
    for all_uav_current_position in UAV_current_Position:
        if all_uav_current_position != UAV_current_Position[current_uav]: #not sure
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
        sum_F_S += copy_value[x_y_not_included_in_other_uav[0]][x_y_not_included_in_other_uav[1]] * \
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
    F_S = F_S_function(x, y, z, UAV_x, UAV_y, UAV_z)
    F_T = F_T_function(x, y, z, UAV_x, UAV_y, UAV_z)
    F_E = F_E_function(x, y, z, UAV_x, UAV_y, UAV_z)
    F = w1 * F_S - w2 * F_T - w3 * F_E
    return F


def residual_energy(UAV_x1, UAV_y1, UAV_z1, UAV_x2, UAV_y2, UAV_z2):
    d_k_m = distance(UAV_x1, UAV_y1, UAV_z1, UAV_x2, UAV_y2, UAV_z2)
    E_k_M = d_k_m / moving_speed * em
    E_k_S_cell = check_coverage(d_u_s_max, theta, UAV_x2, UAV_y2, UAV_z2)
    E_k_S_cell_sum = 0
    E_k_C_sum = 0
    for cell in E_k_S_cell:
        snr1 = SNR(x_uav=UAV_x1, y_uav=UAV_y1, z_uav=UAV_z1, x_c_i=cell[0], y_c_i=cell[1])
        p_g_c_i = 10 ** (-0.7 * math.e ** (0.05 * snr1 - 0.6))
        E_k_S_cell_sum += people_number[cell[0]][cell[1]] * 0.02 * (1 / (1 - p_g_c_i))
        E_k_C_sum += people_number[cell[0]][cell[1]] * (1 / (1 - p_g_c_i)) * ec
    E_k_S = E_k_S_cell_sum * es
    F_E = (E_k_M + E_k_S + E_k_C_sum)
    return F_E


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

def check_connection_with_other_UAVs_in_PSO(x1, y1, z1, x2, y2, z2):
    if (0 < distance(x1,y1,z1,x2,y2,z2) <= d_u_u_max) and (
            all(check_sdpc_function(
                SDPC.coordinate_and_compare_height(x1, y1, x2, y2),x1, y1, z1, x2, y2, z2)) == 1):
        return True


def pso_particle_pos_velocity_initialization(NumParticle, NumDimention, VelocityMax, current_uav):
    Init_PartPosition = [[[] for _ in range(NumDimention)] for _ in range(NumParticle)]
    global GlobalBestValue
    global GlobalBestPosition
    for i in range(NumParticle):
        for j in range(NumDimention):
            PartVelocity[i][j] = random.randint(-VelocityMax, VelocityMax)
        init_value_while_to_false = 0
        while init_value_while_to_false < 2:
            Init_PartPosition[i][0] = random.randint(0, 49)
            Init_PartPosition[i][1] = random.randint(0, 49)
            Init_PartPosition[i][2] = random.uniform(0, 23.9)
            z_cell = z_mountain[Init_PartPosition[i][0]][Init_PartPosition[i][1]]
            if (Init_PartPosition[i][2] > z_cell) and (Init_PartPosition[i] not in prohibited_zone) and (
                    Init_PartPosition[i] not in UAV_current_Position):
                GBS_UAV_connection = []
                for num_gbs_uav in range(NumUAV):
                    if num_gbs_uav != current_uav:
                        func1 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[num_gbs_uav][0],
                            UAV_current_Position[num_gbs_uav][1],
                            UAV_current_Position[num_gbs_uav][2],
                            GBS_position[0], GBS_position[1], GBS_position[2])
                        if func1 == True:
                            GBS_UAV_connection.append(num_gbs_uav)
                if len(GBS_UAV_connection) >= 1:
                    exitFlag = False
                    for connect_to_pso in GBS_UAV_connection:
                        func2 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso][0],
                                    UAV_current_Position[connect_to_pso][1],
                                    UAV_current_Position[connect_to_pso][2],
                                    Init_PartPosition[i][0], Init_PartPosition[i][1],
                                    Init_PartPosition[i][2])
                        if func2 == True:
                            PartPosition[i] = Init_PartPosition[i]
                            PartBestPosition[i] = PartPosition[i]
                            init_value_while_to_false = 5
                            break
                        else:
                            exitFlag2 = False
                            for connect_to_pso2 in range(NumUAV):
                                if (connect_to_pso2 != connect_to_pso) and (connect_to_pso2 != current_uav):
                                    func3 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso][0],
                                                UAV_current_Position[connect_to_pso][1],
                                                UAV_current_Position[connect_to_pso][2],
                                                UAV_current_Position[connect_to_pso2][0],
                                                UAV_current_Position[connect_to_pso2][1],
                                                UAV_current_Position[connect_to_pso2][2])
                                    if func3 == True:
                                        func4 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso2][0],
                                                    UAV_current_Position[connect_to_pso2][1],
                                                    UAV_current_Position[connect_to_pso2][2],
                                                    Init_PartPosition[i][0], Init_PartPosition[i][1],
                                                    Init_PartPosition[i][2])
                                        if func4 == True:
                                            PartPosition[i] = Init_PartPosition[i]
                                            PartBestPosition[i] = PartPosition[i]
                                            init_value_while_to_false = 5
                                            exitFlag = True
                                            break
                                        else:
                                            exitFlag3 = False
                                            for connect_to_pso3 in range(NumUAV):
                                                if (connect_to_pso3 != current_uav) and (
                                                        connect_to_pso3 != connect_to_pso2) and (
                                                        connect_to_pso3 != connect_to_pso):
                                                    func5 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso3][0],
                                                                UAV_current_Position[connect_to_pso3][1],
                                                                UAV_current_Position[connect_to_pso3][2],
                                                                UAV_current_Position[connect_to_pso2][0],
                                                                UAV_current_Position[connect_to_pso2][1],
                                                                UAV_current_Position[connect_to_pso2][2])
                                                    if func5 == True:
                                                        func6 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso3][0],
                                                                    UAV_current_Position[connect_to_pso3][1],
                                                                    UAV_current_Position[connect_to_pso3][2],
                                                                    Init_PartPosition[i][0],
                                                                    Init_PartPosition[i][1],
                                                                    Init_PartPosition[i][2])
                                                        if func6 == True:
                                                            PartPosition[i] = Init_PartPosition[i]
                                                            PartBestPosition[i] = PartPosition[i]
                                                            init_value_while_to_false = 5
                                                            exitFlag = True
                                                            exitFlag2 = True
                                                            break
                                                        else:
                                                            for connect_to_pso4 in range(NumUAV):
                                                                if (connect_to_pso4 != current_uav) and (
                                                                        connect_to_pso4 != connect_to_pso2) and (
                                                                        connect_to_pso4 != connect_to_pso) and (
                                                                        connect_to_pso4 != connect_to_pso3):
                                                                    func7 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[
                                                                                    connect_to_pso3][0],
                                                                                UAV_current_Position[connect_to_pso3][1],
                                                                                UAV_current_Position[connect_to_pso3][2],
                                                                                UAV_current_Position[connect_to_pso4][0],
                                                                                UAV_current_Position[connect_to_pso4][1],
                                                                                UAV_current_Position[connect_to_pso4][2])
                                                                    if func7 == True:
                                                                        func8 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso4][0],
                                                                                    UAV_current_Position[connect_to_pso4][1],
                                                                                    UAV_current_Position[connect_to_pso4][2],
                                                                                    Init_PartPosition[i][0],
                                                                                    Init_PartPosition[i][1],
                                                                                    Init_PartPosition[i][2])
                                                                        if func8 == True:
                                                                            PartPosition[i] = Init_PartPosition[i]
                                                                            PartBestPosition[i] = PartPosition[i]
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
                    func9 = check_connection_with_other_UAVs_in_PSO(Init_PartPosition[i][0],Init_PartPosition[i][1], Init_PartPosition[i][2], GBS_position[0], GBS_position[1],
                        GBS_position[2])
                    if func9 == True:
                        old_uav_to_other_uav_con = []
                        for second_check in range(NumUAV):
                            if second_check != current_uav:
                                func10 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[second_check][0],
                                            UAV_current_Position[second_check][1],
                                            UAV_current_Position[second_check][2],
                                            UAV_current_Position[current_uav][0],
                                            UAV_current_Position[current_uav][1],
                                            UAV_current_Position[current_uav][2])
                                if func10 == True:
                                    old_uav_to_other_uav_con.append(second_check)

                        if (all((distance(UAV_current_Position[con_elem][0],
                                          UAV_current_Position[con_elem][1],
                                          UAV_current_Position[con_elem][2], Init_PartPosition[i][0],
                                          Init_PartPosition[i][1], Init_PartPosition[i][2])) <= d_u_u_max for
                                con_elem in old_uav_to_other_uav_con)) and (all
                            (all(check_sdpc_function(
                            SDPC.coordinate_and_compare_height(
                                UAV_current_Position[con_elem][0],
                                UAV_current_Position[con_elem][1],
                                Init_PartPosition[i][0], Init_PartPosition[i][1]),
                            UAV_current_Position[con_elem][0],
                            UAV_current_Position[con_elem][1],
                            UAV_current_Position[con_elem][2],
                            Init_PartPosition[i][0], Init_PartPosition[i][1],
                            Init_PartPosition[i][2])) == 1 for con_elem in old_uav_to_other_uav_con)):
                            PartPosition[i] = Init_PartPosition[i]
                            PartBestPosition[i] = PartPosition[i]
                            break
        PartBestValue[i] = fitness_function(PartPosition[i][0], PartPosition[i][1], PartPosition[i][2],
                                            UAV_current_Position[current_uav][0],
                                            UAV_current_Position[current_uav][1],
                                            UAV_current_Position[current_uav][2])
        Part_dic[PartBestValue[i]] = PartPosition[i]
    GlobalBestValue = max(key for key in Part_dic.keys())
    GlobalBestPosition = Part_dic[GlobalBestValue]


def pso_iteration(MaxIteration, NumParticle, NumDimention, w_weight, c1, c2, VelocityMax, current_uav):
    Init_PartPosition = [[[] for _ in range(NumDimention)] for _ in range(NumParticle)]
    global GlobalBestValue
    global GlobalBestPosition
    for iter in range(MaxIteration):
        for i in range(NumParticle):
            iter_value_while_to_false = 0
            while iter_value_while_to_false < 2:
                for j in range(NumDimention):
                    r1 = random.random()
                    r2 = random.random()
                    PartVelocity[i][j] = w_weight * PartVelocity[i][j] + c1 * r1 * (
                            PartBestPosition[i][j] - PartPosition[i][j]) + c2 * r2 * (
                                                 GlobalBestPosition[j] - PartPosition[i][j])
                    if PartVelocity[i][j] < -VelocityMax:
                        PartVelocity[i][j] = -VelocityMax
                    if PartVelocity[i][j] > VelocityMax:
                        PartVelocity[i][j] = VelocityMax
                Init_PartPosition[i][0] = round(PartPosition[i][0] + PartVelocity[i][0])
                Init_PartPosition[i][1] = round(PartPosition[i][1] + PartVelocity[i][1])
                Init_PartPosition[i][2] = PartPosition[i][2] + PartVelocity[i][2]
                if (0 <= Init_PartPosition[i][0] <= 49) and (0 <= Init_PartPosition[i][1] <= 49) and (
                        0 <= Init_PartPosition[i][2] <= 24):
                    z_cell = z_mountain[Init_PartPosition[i][0]][Init_PartPosition[i][1]]
                    if (Init_PartPosition[i][2] > z_cell) and (Init_PartPosition[i] not in prohibited_zone) and (
                            Init_PartPosition[i] not in UAV_current_Position):
                        GBS_UAV_connection = []
                        for num_gbs_uav in range(NumUAV):
                            if num_gbs_uav != current_uav:
                                func1 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[num_gbs_uav][0],
                                    UAV_current_Position[num_gbs_uav][1],
                                    UAV_current_Position[num_gbs_uav][2],
                                    GBS_position[0], GBS_position[1], GBS_position[2])
                                if func1 == True:
                                    GBS_UAV_connection.append(num_gbs_uav)
                        if len(GBS_UAV_connection) >= 1:
                            exitFlag = False
                            for connect_to_pso in GBS_UAV_connection:
                                func2 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso][0],
                                            UAV_current_Position[connect_to_pso][1],
                                            UAV_current_Position[connect_to_pso][2],
                                            Init_PartPosition[i][0], Init_PartPosition[i][1],
                                            Init_PartPosition[i][2])
                                if func2 == True:
                                    PartPosition[i] = Init_PartPosition[i]
                                    iter_value_while_to_false = 5
                                    break
                                else:
                                    exitFlag2 = False
                                    for connect_to_pso2 in range(NumUAV):
                                        if (connect_to_pso2 != connect_to_pso) and (connect_to_pso2 != current_uav):
                                            func3 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso][0],
                                                        UAV_current_Position[connect_to_pso][1],
                                                        UAV_current_Position[connect_to_pso][2],
                                                        UAV_current_Position[connect_to_pso2][0],
                                                        UAV_current_Position[connect_to_pso2][1],
                                                        UAV_current_Position[connect_to_pso2][2])
                                            if func3 == True:
                                                func4 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso2][0],
                                                            UAV_current_Position[connect_to_pso2][1],
                                                            UAV_current_Position[connect_to_pso2][2],
                                                            Init_PartPosition[i][0], Init_PartPosition[i][1],
                                                            Init_PartPosition[i][2])
                                                if func4 == True:
                                                    PartPosition[i] = Init_PartPosition[i]
                                                    iter_value_while_to_false = 5
                                                    exitFlag = True
                                                    break
                                                else:
                                                    exitFlag3 = False
                                                    for connect_to_pso3 in range(NumUAV):
                                                        if (connect_to_pso3 != current_uav) and (
                                                                connect_to_pso3 != connect_to_pso2) and (
                                                                connect_to_pso3 != connect_to_pso):
                                                            func5 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso3][0],
                                                                        UAV_current_Position[connect_to_pso3][1],
                                                                        UAV_current_Position[connect_to_pso3][2],
                                                                        UAV_current_Position[connect_to_pso2][0],
                                                                        UAV_current_Position[connect_to_pso2][1],
                                                                        UAV_current_Position[connect_to_pso2][
                                                                            2])
                                                            if func5 == True:
                                                                func6 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso3][0],
                                                                            UAV_current_Position[connect_to_pso3][1],
                                                                            UAV_current_Position[connect_to_pso3][2],
                                                                            Init_PartPosition[i][0],
                                                                            Init_PartPosition[i][1],
                                                                            Init_PartPosition[i][2])
                                                                if func6 == True:
                                                                    PartPosition[i] = Init_PartPosition[i]
                                                                    iter_value_while_to_false = 5
                                                                    exitFlag = True
                                                                    exitFlag2 = True
                                                                    break
                                                                else:
                                                                    for connect_to_pso4 in range(NumUAV):
                                                                        if (connect_to_pso4 != current_uav) and (
                                                                                connect_to_pso4 != connect_to_pso2) and (
                                                                                connect_to_pso4 != connect_to_pso) and (
                                                                                connect_to_pso4 != connect_to_pso3):
                                                                            func7 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[
                                                                                            connect_to_pso3][0],
                                                                                        UAV_current_Position[connect_to_pso3][1],
                                                                                        UAV_current_Position[connect_to_pso3][2],
                                                                                        UAV_current_Position[connect_to_pso4][0],
                                                                                        UAV_current_Position[connect_to_pso4][1],
                                                                                        UAV_current_Position[connect_to_pso4][2])
                                                                            if func7 == True:
                                                                                func8 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[connect_to_pso4][0],
                                                                                            UAV_current_Position[connect_to_pso4][1],
                                                                                            UAV_current_Position[connect_to_pso4][2],
                                                                                            Init_PartPosition[i][0],
                                                                                            Init_PartPosition[i][1],
                                                                                            Init_PartPosition[i][2])
                                                                                if func8 == True:
                                                                                    PartPosition[i] = Init_PartPosition[i]
                                                                                    iter_value_while_to_false = 5
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
                            func9 = check_connection_with_other_UAVs_in_PSO(Init_PartPosition[i][0],
                                Init_PartPosition[i][1], Init_PartPosition[i][2], GBS_position[0], GBS_position[1],
                                GBS_position[2])
                            if func9 == True:
                                old_uav_to_other_uav_con = []
                                for second_check in range(NumUAV):
                                    if second_check != current_uav:
                                        func10 = check_connection_with_other_UAVs_in_PSO(UAV_current_Position[second_check][0],
                                                    UAV_current_Position[second_check][1],
                                                    UAV_current_Position[second_check][2],
                                                    UAV_current_Position[current_uav][0],
                                                    UAV_current_Position[current_uav][1],
                                                    UAV_current_Position[current_uav][2])
                                        if func10 == True:
                                            old_uav_to_other_uav_con.append(second_check)
                                if (all((distance(UAV_current_Position[con_elem][0],
                                                  UAV_current_Position[con_elem][1],
                                                  UAV_current_Position[con_elem][2], Init_PartPosition[i][0],
                                                  Init_PartPosition[i][1], Init_PartPosition[i][2])) <= d_u_u_max for
                                        con_elem in old_uav_to_other_uav_con)) and (all
                                    (all(check_sdpc_function(
                                    SDPC.coordinate_and_compare_height(
                                        UAV_current_Position[con_elem][0],
                                        UAV_current_Position[con_elem][1],
                                        Init_PartPosition[i][0], Init_PartPosition[i][1]),
                                    UAV_current_Position[con_elem][0],
                                    UAV_current_Position[con_elem][1],
                                    UAV_current_Position[con_elem][2],
                                    Init_PartPosition[i][0], Init_PartPosition[i][1],
                                    Init_PartPosition[i][2])) == 1 for con_elem in old_uav_to_other_uav_con)):
                                    PartPosition[i] = Init_PartPosition[i]
                                    break
            fit_func = fitness_function(PartPosition[i][0], PartPosition[i][1], PartPosition[i][2],
                                        UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                                        UAV_current_Position[current_uav][2])
            if fit_func > PartBestValue[i]:
                PartBestValue[i] = fit_func
                for j in range(NumDimention):
                    PartBestPosition[i][j] = PartPosition[i][j]
            if fit_func > GlobalBestValue:
                GlobalBestValue = fit_func
                GlobalBestPosition = PartPosition[i]

global_average_f = 0
global_average_ft = 0
global_average_fs = 0
global_average_fe = 0
global_time = 0
average_accumulated_total_value = 0
average_accumulated_time_value= 0
average_accumulated_energy_value = 0
average_accumulated_sensing_value = 0
average_average_sensing_value = 0
average_max_time = 0
average_number_of_movement = 0
average_number_of_covered_cells = 0

simulation_time = 1
while simulation_time < 21:
    plot_sensing = []
    plot_energy = []
    plot_uav_num = []
    plot_time = []
    plot_uav_num_time = []

    # start_time = time.time()
    NumUAV =5
    MAX_time = 15
    MIN_energy = 20
    energy_consumption = [400 for _ in range(NumUAV)]

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
    # w1, w2, w3 = 0.4, 0.1, 0.5

    R = 3.5 # 120 secs to 2 mins
    NOR_S = 124
    NOR_T = 43.9
    NOR_E = 75
    # NOR_T = 157.23
    # NOR_E = 33.27
    em = 2
    es = 1
    ec = 0.03
    a_coef = (ValueMax - ValueMin) / ((math.e ** R) - 1)
    b_coef = ValueMin - a_coef

    # PSO parameters:
    NumParticle = 5
    NumDimention = 3
    MaxIteration = 50
    VelocityMax = 7
    c1, c2 = 2, 2
    w_weight = 0.5
    GBS_position = [0, 0, 0]

    v_value = np.ones((50, 50), float)
    v_value.fill(2.0)
    prev_time = np.zeros((50, 50), float)
    prev_time.fill(100)
    sensor_num = np.zeros((50, 50), int)
    number_of_sensor_function()
    UAV_current_Position = [[0, 0, 0] for _ in range(NumUAV)]

    time_of_each_UAV = [0 for _ in range(NumUAV)]
    copy_time_of_each_UAV = [0 for _ in range(NumUAV)]
    uav_terminates = []
    # vsi_x = 15
    # vsi_y = 15
    # time_list = [0]
    # value_list = [2]
    average_f_value = 0
    accumulative_sensing_value_list = []
    accumulative_total_value_list = []
    average_s_value = 0
    average_t_value = 0
    average_e_value = 0
    movement_uav = 0
    movement_uav_list = []
    covered_cell_check = np.zeros((50, 50), float)
    accumulated_covered_cell = []
    movement = 0
    old_current_uav = 6
    list_to_simulate = []
    while movement < 1:
        # if movement_uav > 20:
        #     n_components = 3
        #     X_surface, truth_surface = make_blobs(n_samples=300, centers=[[0.72, 0.72], [0.6, 1.4], [1.4, 1.20]],
        #                                           cluster_std=[0.24, 0.12, 0.18], random_state=42)
        #     x_surface = X_surface[:, 0]
        #     y_surface = X_surface[:, 1]
        #     xx_surface, yy_surface = np.mgrid[-0.02:1.98:51j, -0.02:1.98:51j]
        #     positions_surface = np.vstack([xx_surface.ravel(), yy_surface.ravel()])
        #     values_surface = np.vstack([x_surface, y_surface])
        #     kernel_surface = st.gaussian_kde(values_surface)
        #     f_surface = np.reshape(kernel_surface(positions_surface).T, xx_surface.shape)
        #     zz_surface = f_surface * 0.35
        #
        #     X_sensor, y_true_sensor = make_blobs(n_samples=20000,
        #                                          centers=[[0.44, 0.44], [0.8, 0.8], [0.6, 1.52], [1.4, 0.6],
        #                                                   [1.52, 1.4]],
        #                                          cluster_std=[0.16, 0.16, 0.204, 0.21, 0.2], random_state=0)
        #     x_sensors = X_sensor[:, 0]
        #     y_sensors = X_sensor[:, 1]
        #     xx_sensors, yy_sensors = np.mgrid[0:1.96:50j, 0:1.96:50j]
        #     positions_sensors = np.vstack([xx_sensors.ravel(), yy_sensors.ravel()])
        #     values_sensor = np.vstack([x_sensors, y_sensors])
        #     kernel_sensor = st.gaussian_kde(values_sensor)
        #     f_sensor = np.reshape(kernel_sensor(positions_sensors).T, xx_sensors.shape)
        #     zz_sensor = f_sensor * 34.41  # 20 000 sensors
        #     vector_sensor = np.vectorize(np.int32)
        #     zz_sensor = vector_sensor(zz_sensor)
        #     fig, ax = plt.subplots(nrows=1, ncols=1)
        #     ax.set_xlim(xmin=0, xmax=2)
        #     ax.set_ylim(ymin=0, ymax=2)
        #     ax.xaxis.set_major_locator(MultipleLocator(base=0.2))
        #     ax.yaxis.set_major_locator(MultipleLocator(base=0.2))
        #     plt.contour(xx_surface, yy_surface, zz_surface, 20, cmap='RdGy')
        #     plt.scatter(x_sensors, y_sensors, marker='+', s=3, color='grey', zorder=3)
        #     # ax.set_title('Contour map')
        #     ax.set_xlabel('x [km] ', size=15)
        #     ax.set_ylabel('y [km] ', size=15)
        #     plt.plot(0, 0, marker='.', color='brown', markersize=26)
        #     ax.text(0.02, 0.02, "GBS", color='brown', size=17)
        #     ax.tick_params(which='major', length=5, width=2, direction='in', bottom=True, top=True, left=True,
        #                    right=True)
        #     ax.grid(linewidth=1, color='lightgrey')
        #
        #
        #     for uav_connection in range(NumUAV):
        #         if uav_connection == 0:
        #             color = 'red'
        #         elif uav_connection == 1:
        #             color = 'green'
        #         elif uav_connection == 2:
        #             color = 'magenta'
        #         elif uav_connection == 3:
        #             color = 'purple'
        #         elif uav_connection == 4:
        #             color = 'navy'
        #         plt.scatter(UAV_current_Position[uav_connection][0] * 0.04, UAV_current_Position[uav_connection][1] * 0.04, s=100,
        #                     zorder=10, color=color, label=f"UAV {uav_connection+1} " )
        #
        #         for con_num in range(NumUAV):
        #             if con_num != uav_connection:
        #                 if distance(UAV_current_Position[uav_connection][0] , UAV_current_Position[uav_connection][1],
        #                             UAV_current_Position[uav_connection][2],
        #                             UAV_current_Position[con_num][0], UAV_current_Position[con_num][1], UAV_current_Position[con_num][2]) <= d_u_u_max:
        #                     cell_j = SDPC.coordinate_and_compare_height(UAV_current_Position[uav_connection][0],
        #                                                                 UAV_current_Position[uav_connection][1],
        #                                                                 UAV_current_Position[con_num][0], UAV_current_Position[con_num][1])
        #                     sdpc_list_check = check_sdpc_function(cell_j, UAV_current_Position[uav_connection][0],
        #                                                           UAV_current_Position[uav_connection][1],
        #                                                           UAV_current_Position[uav_connection][2],
        #                                                           UAV_current_Position[con_num][0], UAV_current_Position[con_num][1], UAV_current_Position[con_num][2])
        #                     if all(sdpc_list_check) == 1:
        #                         plt.plot([UAV_current_Position[uav_connection][0] * 0.04, UAV_current_Position[con_num][0] * 0.04],
        #                                  [UAV_current_Position[uav_connection][1] * 0.04,  UAV_current_Position[con_num][1] * 0.04],
        #                                  color='blue', zorder=8, linestyle='solid', linewidth = 1)
        #
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
        #                 plt.plot([UAV_current_Position[uav_connection][0] * 0.04, 0],
        #                          [UAV_current_Position[uav_connection][1]* 0.04, 0],
        #                          color='blue', zorder=8, linestyle='solid', linewidth = 1)
        #         # handles, labels = plt.gca().get_legend_handles_labels()
        #         # line = Line2D([0], [0], label='Connection line', color='blue')
        #         # handles.extend([line])
        #         # plt.legend(handles=handles, loc = 'upper left'lower right)
        #         plt.legend(fontsize=12, loc = 'center left')
        # if movement_uav == 3:
        #     movement = 3
        GlobalBestValue = -5
        PartVelocity = [[[] for _ in range(NumDimention)] for _ in range(NumParticle)]
        PartPosition = [[[] for _ in range(NumDimention)] for _ in range(NumParticle)]
        PartBestPosition = [[[] for _ in range(NumDimention)] for _ in range(NumParticle)]
        PartBestValue = [[] for _ in range(NumParticle)]
        Part_dic = dict()
        GlobalBestPosition = []

        if all(time_of_each_UAV) != 0:
            copy_time_of_each_UAV2 = []
            for more_zero in copy_time_of_each_UAV:
                if more_zero != 0:
                    copy_time_of_each_UAV2.append(more_zero)
            min_time = min(every_time for every_time in copy_time_of_each_UAV2)
            current_uav = time_of_each_UAV.index(min_time)
        else:
            current_uav = time_of_each_UAV.index(0)


        if old_current_uav < NumUAV:
            for x_value in range(50):
                for y_value in range(50):
                    if abs(time_of_each_UAV[old_current_uav] - prev_time[x_value][y_value]) <= R:
                        v_value[x_value][y_value] = a_coef * math.exp(
                            time_of_each_UAV[old_current_uav] - prev_time[x_value][y_value]) + b_coef
                    else:
                        v_value[x_value][y_value] = ValueMax
            check_coverage_old_uav = check_coverage(d_u_s_max, theta, UAV_current_Position[old_current_uav][0],
                                                             UAV_current_Position[old_current_uav][1],
                                                             UAV_current_Position[old_current_uav][2])
            for cov in check_coverage_old_uav:
                v_value[cov[0]][cov[1]] = 0.2
                prev_time[cov[0]][cov[1]] = time_of_each_UAV[old_current_uav]
            UAV_current_Position[old_current_uav] = [0, 0, 0]
            old_current_uav = 6


        for x_value in range(50):
            for y_value in range(50):
                if abs(time_of_each_UAV[current_uav] - prev_time[x_value][y_value]) <= R:
                    v_value[x_value][y_value] = a_coef * math.exp(time_of_each_UAV[current_uav] - prev_time[x_value][y_value]) + b_coef
                else:
                    v_value[x_value][y_value] = ValueMax
        # if movement_uav == 0:
        #     x_y_vsi = check_coverage_new_position[0]
        #     vsi_x = x_y_vsi[0]
        #     vsi_y = x_y_vsi[1]
        # value_list.append(v_value[vsi_x][vsi_y])
        # time_list.append(time_of_each_UAV[current_uav] + moving_time)
        check_coverage_current_position = check_coverage(d_u_s_max, theta, UAV_current_Position[current_uav][0],UAV_current_Position[current_uav][1], UAV_current_Position[current_uav][2])
        sensor_number_in_coverage = 0
        for cov in check_coverage_current_position:
            v_value[cov[0]][cov[1]] = 0.2
            prev_time[cov[0]][cov[1]] = time_of_each_UAV[current_uav]
            sensor_number_in_coverage += people_number[cov[0]][cov[1]]
        # value_list.append(v_value[vsi_x][vsi_y])
        # time_list.append(time_of_each_UAV[current_uav] + moving_time)


        if sensor_number_in_coverage < 5:
            pso_particle_pos_velocity_initialization(NumParticle, NumDimention, VelocityMax, current_uav)
            pso_iteration(MaxIteration, NumParticle, NumDimention, w_weight, c1, c2, VelocityMax, current_uav)
            next_uav_position = GlobalBestPosition
        else:
            next_uav_position = UAV_current_Position[current_uav]
        moving_time = (distance(UAV_current_Position[current_uav][0],UAV_current_Position[current_uav][1], UAV_current_Position[current_uav][2], next_uav_position[0], next_uav_position[1], next_uav_position[2]) / moving_speed) / 60
        F_S = F_S_function(next_uav_position[0], next_uav_position[1], next_uav_position[2],
                           UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                           UAV_current_Position[current_uav][2])
        average_s_value += F_S
        accumulative_sensing_value_list.append(average_s_value)
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
        accumulative_total_value_list.append(average_f_value)
        check_coverage_new_position = check_coverage(d_u_s_max, theta, next_uav_position[0],next_uav_position[1],next_uav_position[2])

        for x_value in range(50):
            for y_value in range(50):
                if abs((time_of_each_UAV[current_uav] + moving_time) - prev_time[x_value][y_value]) <= R:
                    v_value[x_value][y_value] = a_coef * math.exp((time_of_each_UAV[current_uav] + moving_time ) - prev_time[x_value][y_value]) + b_coef
                else:
                    v_value[x_value][y_value] = ValueMax
        new_position_stay_time_sum = 0
        for cov_next in check_coverage_new_position:
            covered_cell_check[cov_next[0]][cov_next[1]] = 1
            snr = SNR(UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1],
                      UAV_current_Position[current_uav][2], cov_next[0], cov_next[1])
            p_g_c_i = 10 ** (-0.7 * math.e ** (0.05 * snr - 0.6))
            next_stay_time_c_i = people_number[cov_next[0]][cov_next[1]] * 0.02 * 1 / (1 - p_g_c_i)
            new_position_stay_time_sum += next_stay_time_c_i
        new_position_stay_time = new_position_stay_time_sum / 60

        num_covered = 0
        for ii in range(50):
            for jj in range(50):
                if covered_cell_check[ii][jj] == 1:
                    num_covered += 1
        accumulated_covered_cell.append(num_covered)

        time_update_in_movement = time_of_each_UAV[current_uav] + moving_time + new_position_stay_time
        # if movement_uav > 20:
        #     list_to_simulate.append(time_update_in_movement)
        copy_time_of_each_UAV[current_uav] = time_update_in_movement
        time_of_each_UAV[current_uav] = time_update_in_movement
        plot_time.append(time_of_each_UAV[current_uav])


        residual_e = residual_energy(UAV_current_Position[current_uav][0], UAV_current_Position[current_uav][1], UAV_current_Position[current_uav][2],next_uav_position[0], next_uav_position[1], next_uav_position[2] )
        energy_consumption[current_uav] = energy_consumption[current_uav] - residual_e

        if (energy_consumption[current_uav] < MIN_energy) or (time_of_each_UAV[current_uav] > MAX_time):
            copy_time_of_each_UAV[current_uav] = 0
            uav_terminates.append(current_uav)
            print(f"there will be minus 1 uav {current_uav}")
            # UAV_current_Position[current_uav] = [0, 0, 0]
            plot_uav_num_time.append(time_of_each_UAV[current_uav])
            old_current_uav = current_uav
        UAV_current_Position[current_uav] = next_uav_position


        # while_close = []
        # for i_j in copy_time_of_each_UAV:
        #     if i_j == 0:
        #         while_close.append(i_j)

        if len(uav_terminates) == NumUAV:
            print(f"There is no UAV at movement time ")
            movement = 10
        # print(f'sensing { F, F_S, F_T, F_E}')
        # print(f"current UAV number {current_uav}: current position {next_uav_position}")
        # print(movement_uav)
        movement_uav += 1
    # print(f"accumulative total fitness value{average_f_value}")
    # print(f"average sensing, time, energy {average_s_value/movement_uav, average_t_value/ movement_uav, average_e_value/ movement_uav}")
    # print("max sensing, time, energy value", average_s_value, average_t_value, average_e_value)
    # print(f"Num of movements {movement_uav}")
    # print(f"Max number of covered cells {max(accumulated_covered_cell)}")
    # print(f"plot last uav terminated its work {max(plot_uav_num_time)}")
    # print(f"plot uav terminated its work {plot_uav_num_time}")
    # #
    # print(f" time  {plot_time}")
    # print(f"accumulateve total value{accumulative_total_value_list}")
    # print(f" accumulative sensing value {accumulative_sensing_value_list}")
    # print(f"accumulated covered cells {accumulated_covered_cell}")
    average_number_of_movement += movement_uav
    average_accumulated_total_value += average_f_value
    average_accumulated_sensing_value += average_s_value
    average_accumulated_time_value += average_t_value
    average_accumulated_energy_value += average_e_value
    average_average_sensing_value += average_s_value / movement_uav
    average_number_of_covered_cells += max(accumulated_covered_cell)
    average_max_time += max(plot_time)
    print("max total value", average_accumulated_total_value / simulation_time)
    print("max sensing value", average_accumulated_sensing_value / simulation_time)
    print("max time value", average_accumulated_time_value / simulation_time)
    print("max energy value", average_accumulated_energy_value / simulation_time)

    print(f"Num of movements {average_number_of_movement / simulation_time}")
    print(f"Max number of covered cells {average_number_of_covered_cells / simulation_time}")
    print(f" time  {average_max_time / simulation_time}")
    print(f" average sensing value {average_average_sensing_value / simulation_time}")
    simulation_time += 1
    print("simulation:", simulation_time)
# print("max total value", average_accumulated_total_value / 20)
# print("max sensing value", average_accumulated_sensing_value / 20)
# print("max time value", average_accumulated_time_value / 20)
# print("max energy value", average_accumulated_energy_value / 20)
#
# print(f"Num of movements {average_number_of_movement / 20}")
# print(f"Max number of covered cells {average_number_of_covered_cells / 20}")
# print(f" time  {average_max_time / 20}")
# print(f" average sensing value {average_average_sensing_value / 20}")

# print("max sensing value", average_accumulated_sensing_value / 3)
# print(f"Num of movements {average_number_of_movement/3}")
# print(f"Max number of covered cells {average_number_of_covered_cells/3}")
# print(f" time  {average_max_time / 3}")
# print(f" average sensing value {average_average_sensing_value / 3}")

# fig4 = plt.figure(30)
# ax = plt.axes()
# plt.plot(time_list, value_list)
# ax.set(xlabel = 'Time t in minutes',ylabel = 'VSI')
# ax.set_ylim(ymin = 0)
# ax.yaxis.set_major_locator(MultipleLocator(base=0.2))
# print("time list ", time_list)
# print("value list", value_list)
# for jk in v_value:
#     print(jk)
#
# plt.show()