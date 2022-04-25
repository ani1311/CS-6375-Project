import csv
import numpy as np
import matplotlib.pyplot as plt

csv_log_file = None
csv_log_writer = None
fieldnames = ['episode', 'reward']


def get_filename_without_extension(model_name, params):
    file_name = "logs/" + model_name
    for k, v in params.items():
        file_name = file_name + "_" + str(k) + "=" + str(v)
    return file_name


def save_heat_map_mc(model, params, state_action, n):
    g = []
    for i in range(n):
        t = []
        for j in range(n):
            t.append(-9999.0)
        g.append(t)

    for k, v in state_action.items():
        g[k[0][0]][k[0][1]] = max(g[k[0][0]][k[0][1]], v[0])

    for i in range(n):
        for j in range(n):
            if(g[i][j] == -9999.0):
                g[i][j] = 0

    a = np.array(g)
    out_file_name = get_filename_without_extension(model, params) + ".png"
    print("Saving head map for", out_file_name)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.savefig(out_file_name)


def create_file(model_name, params):
    global csv_log_writer, csv_log_file
    csv_log_file = open(get_filename_without_extension(
        model_name, params) + ".csv", mode='w')
    csv_log_writer = csv.writer(csv_log_file, delimiter=',')
    csv_log_writer.writerow(fieldnames)


def write_entry(episode, reward):
    global csv_log_writer, csv_log_file
    csv_log_writer.writerow([episode, reward])


def save_file():
    global csv_log_writer, csv_log_file
    csv_log_file.close()


def save_heat_map_td(model, params, state_action, n):
    g = []
    for i in range(n):
        t = []
        for j in range(n):
            t.append(-9999.0)
        g.append(t)

    for i in range(n):
        for j in range(n):
            for k, v in state_action.items():
                if k[0] == (i, j):
                    g[i][j] = max(g[i][j], v)

    for i in range(n):
        for j in range(n):
            if(g[i][j] == -9999.0):
                g[i][j] = 0

    a = np.array(g)
    out_file_name = get_filename_without_extension(model, params) + ".png"
    print("Saving head map for", out_file_name)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.savefig(out_file_name)
