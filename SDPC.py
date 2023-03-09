x_cor_list = []
y_cor_list = []

def coordinate_and_compare_height(x1, y1, x2, y2):
    global x_cor_list
    global y_cor_list

    x_cor = []
    y_cor = []
    y_cor2 = []
    x_cor2 = []


    if x1 > x2 and y1 > y2:
        for i in range(x1, x2 - 1, -1):
            x_cor.append(i)
        for j in range(y1, y2 - 1, -1):
            y_cor.append(j)
    elif x1 > x2 and y1 < y2:
        for i in range(x1, x2 - 1, -1):
            x_cor.append(i)
        for j in range(y1, y2 + 1):
            y_cor.append(j)
    elif x1 < x2 and y1 > y2:
        for i in range(x1, x2 + 1):
            x_cor.append(i)
        for j in range(y1, y2 - 1, -1):
            y_cor.append(j)
    elif x1 < x2 and y1 < y2:
        for i in range(x1, x2 + 1):
            x_cor.append(i)
        for j in range(y1, y2 + 1):
            y_cor.append(j)
    elif x1 == x2 and y1 > y2:
        x_cor.append(x1)
        for j in range(y1, y2 - 1, -1):
            y_cor.append(j)
    elif x1 == x2 and y1 < y2:
        x_cor.append(x1)
        for j in range(y1, y2 + 1):
            y_cor.append(j)
    elif x1 > x2 and y1 == y2:
        y_cor.append(y1)
        for i in range(x1, x2 - 1, -1):
            x_cor.append(i)
    elif x1 < x2 and y1 == y2:
        y_cor.append(y1)
        for i in range(x1, x2 + 1):
            x_cor.append(i)
    elif x1 == x2 and y1 == y2:
        y_cor.append(x1)
        x_cor.append(y1)
    #
    # print(x_cor, "x coord", len(x_cor))
    # print(y_cor, "y coord", len(y_cor))
    if len(x_cor) > len(y_cor):
        coef = int(len(x_cor) / len(y_cor))
        for y_i in y_cor:
            for i in range(coef):
                y_cor2.append(y_i)
        if len(y_cor2) < len(x_cor):
            for k in range(len(x_cor) - len(y_cor2)):
                y_cor2.append(y_cor[-1])
        y_cor = y_cor2
    elif len(x_cor) < len(y_cor):
        coef = int(len(y_cor) / len(x_cor))
        for x_i in x_cor:
            for i in range(coef):
                x_cor2.append(x_i)
        if len(x_cor2) < len(y_cor):
            for k in range(len(y_cor) -  len(x_cor2)):
                x_cor2.append(x_cor[-1])
        x_cor = x_cor2
    x_cor = x_cor[1: -1]
    y_cor = y_cor[1: -1]
    x_cor_list = x_cor
    y_cor_list = y_cor
    cell_axes = []
    for a, b in zip(x_cor_list, y_cor_list):
        cell_axes.append([a, b])
    return cell_axes
    # print(x_cor, "x coord", len(x_cor))
    # print(y_cor, "y coord", len(y_cor))
#
# c = coordinate_and_compare_height(38,22,37,15)
# for i in c:
#     print(i)

# print(len(x_cor_list))
# print(y_cor_list)
# print(x_cor_list)
# coordinate_and_compare_height(38,22,20,19)
# print(len(x_cor_list))
# print(y_cor_list)
# print(x_cor_list)