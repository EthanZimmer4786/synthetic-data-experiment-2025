import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

import sqlite3

########## Pull Data ##########

def pull_data(path, *args):
    """Returns a list of data with each index representing the column of the same name as the provided arguments in the fucntion
       
       Example:

       x, y, z = pull_data(path, 'x_data', 'y_data', 'z_data')"""
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    data_arr = []
    for arg in args:
        data = [x[0] for x in cursor.execute(f"SELECT {arg} FROM data")]
        
        # if type(data) == str and data[0] == '[' and data[-1] == ']':
        #     data = eval(data)
        #     print('Evaluated stringified list')

        data_arr.append(data)

    conn.close()

    return data_arr

########## Reduce List ##########

def split_list_xy(x, y):
    """
    Split x list into tuples containing a list and value of y.
    The list in the tuple contains x values that share a similar value of y, compared by index.
    """

    arr = []
    for i in range(len(x)):
        arr.append((x[i], y[i]))

    split_arr = []
    for i in range(len(arr)):
        if (arr[i][1]) not in [(split[1]) for split in split_arr]:
            split_arr.append(([], arr[i][1]))
        
        split_arr[[(split[1]) for split in split_arr].index((arr[i][1]))][0].append(arr[i][0])

    return split_arr

def split_list_xyz(x, y, z):
    """
    Split x list into tuples containing a list and value of y.
    The list in the tuple contains x values that share a similar value of y, compared by index.
    """

    arr = []
    for i in range(len(x)):
        arr.append((x[i], y[i], z[i]))

    split_arr = []
    for i in range(len(arr)):
        if (arr[i][1], arr[i][2]) not in [(split[1], split[2]) for split in split_arr]:
            split_arr.append(([], arr[i][1], arr[i][2]))
        
        split_arr[[(split[1], split[2]) for split in split_arr].index((arr[i][1], arr[i][2]))][0].append(arr[i][0])
    
    return split_arr

########## Heatmap ##########

def heatmap(fig, relative_cmap, log_scale, mode=0):

    x_axis = 'train_image_count'
    y_axis = 'fake_image_ratio'
    z_axis = 'test_acc'

    x, y, z = pull_data(DATABASE_PATH, x_axis, y_axis, z_axis)

    levels = 50

    x2 = []
    y2 = []
    z2 = []
    if mode == 0:
        for row in split_list_xyz(z, x, y):
            x2.append(row[1])
            y2.append(row[2])
            z2.append(np.average(row[0]))

    elif mode == 1:
        for row in split_list_xyz(z, x, y):
            x2.append(max(50, row[1] * (1 - row[2])))
            y2.append(max(50, row[1] * row[2]))
            z2.append(np.average(row[0]))

    cmap = plt.get_cmap('viridis')

    if relative_cmap == False:
        levels = list(range(levels + 1))
        for i in range(len(levels)):
            levels[i] = levels[i] / levels[-1]
    
    ax = fig.add_subplot()

    ax.tricontour(x2, y2, z2, levels=levels, linewidths=0.1, colors='k')
    contourf = ax.tricontourf(x2, y2, z2, levels=levels, cmap=cmap)
    ax.scatter(x2, y2, c='k', marker='.', s=5)

    if(log_scale):
        ax.set_xscale('log')
        ax.set_xticks(ticks=LOG_TICKS, labels=[str(x) for x in LOG_TICKS])
        ax.set_xlim(xmin=50)
        
        if mode == 1:
            ax.set_yscale('log')
            ax.set_yticks(ticks=LOG_TICKS, labels=[str(x) for x in LOG_TICKS])
            ax.set_ylim(ymin=50)
    
    if mode == 0:
        ax.set(title=DATABASE_PATH, xlabel=x_axis, ylabel=y_axis)
    elif mode ==1:
        ax.set(title=DATABASE_PATH, xlabel="real_image_count", ylabel="fake_image_count")

    fig.colorbar(contourf, ax=ax, label=z_axis)

    return fig

########## Trial History Graph ##########

def trial_history_graph(fig):

    x_axis = 'epochs'
    y_axis = 'val_acc_arr'
    z_axis = 'train_image_count'

    x, y, z = pull_data(DATABASE_PATH, x_axis, y_axis, z_axis)
    y = [eval(y) for y in y] # TODO Error

    maxim = max(z)
    z = [i / maxim for i in z]

    cmap = plt.get_cmap('viridis')

    ax = fig.add_subplot()

    for i in range(len(x)):
        ax.plot(range(x[i]), y[i], c=cmap(z[i]))

    ax.set(title=DATABASE_PATH, xlabel=x_axis, ylabel=y_axis)
    
    # plt.colorbar(cmap, label=z_axis)

    return fig

########## Trial Variability Comparison ##########

def trial_variability_comparison(fig):

    x_axis='train_image_count'
    y_axis='test_acc'
    z_axis='fake_image_ratio'

    x, y, z = pull_data(DATABASE_PATH, x_axis, y_axis, z_axis)

    x_set = sorted(list(set(x)))

    y_low_z = [([], x) for x in x_set]
    y_high_z = [([], x) for x in x_set]

    for i in range(len(z)):
        if z[i] == min(z):
            y_low_z[x_set.index(x[i])][0].append(y[i])
        if z[i] == max(z):
            y_high_z[x_set.index(x[i])][0].append(y[i])
    
    y_low_z = [sum(i[0]) / len(i[0]) for i in y_low_z]
    y_high_z = [sum(i[0]) / len(i[0]) for i in y_high_z]
        
    cmap = plt.get_cmap('viridis')

    width = 0.3

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    scatter = ax1.scatter(x, y, c=cmap(z), marker='.', s=20)

    ax1.set_xscale('log')
    ax1.set_xticks(ticks=LOG_TICKS, labels=[str(x) for x in LOG_TICKS])

    ax1.set(title='Variation in Test Accuracy as # of Images Increases', xlabel=x_axis, ylabel=y_axis)

    low_z_bar = ax2.bar([x - (width / 2) for x in range(len(x_set))], y_low_z, width, color=cmap(0.3))
    high_z_bar = ax2.bar([x + (width / 2) for x in range(len(x_set))], y_high_z, width, color=cmap(0.7))
    
    ax2.set(xlabel=x_axis, ylabel=y_axis)
    ax2.set_xticks(range(len(x_set)), labels=[str(x) for x in x_set])

    ax2.legend([low_z_bar, high_z_bar], ['Real (0.0 ratio)', 'Synthetic (1.0 ratio)'])
    
    fig.colorbar(scatter, cmap=cmap, label=z_axis)

    fig.tight_layout()
    return fig

########## Compute Time Graph ##########

def compute_time_graph(fig):

    x_axis =  'train_image_count'
    y1_axis = 'test_acc'
    y2_axis = 'execution_time_sec'

    x, y1, y2 = pull_data(DATABASE_PATH, x_axis, y1_axis, y2_axis)

    ax1 = fig.add_subplot()
    
    y1 = split_list_xy(y1, x)
    y1 = [sum(i[0]) / len(i[0]) for i in y1]
    y2 = split_list_xy(y2, x)
    y2 = [sum(i[0]) / len(i[0]) for i in y2]

    x = sorted(list(set(x)))

    ax1.set(title='Test Accuracy and Computation Time Related to # of Images', xlabel=x_axis, ylabel=y1_axis)
    ax1.set_xscale('log')
    ax1.set_xticks(ticks=LOG_TICKS, labels=[str(x) for x in LOG_TICKS])
    ax1.plot(x, y1, color='k')
    ax1.set_ylim(ymin=0)

    ax2 = ax1.twinx()  # Instantiate a second Axes that shares the same x-axis

    ax2.set_ylabel(y2_axis, color='tab:blue')
    ax2.tick_params('y', labelcolor='tab:blue')
    ax2.plot(x, y2, color='tab:blue')
    ax2.set_ylim(ymin=0) 

    return fig

########## Deviation Graph ##########

def deviation_graph(fig):
    x_axis = 'trial'
    y_axis = 'test_acc'
    z_axis = 'fake_image_ratio'
    axes_split = 'train_image_count'

    y, z, split = pull_data(DATABASE_PATH, y_axis, z_axis, axes_split)

    split_set = sorted(list(set(split)))

    y_split = split_list_xyz(y, split, z)

    cmap = plt.get_cmap('viridis')

    norm = plt.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  

    axes = []
    for ax in range(len(split_set)):
        axes.append(fig.add_subplot(round(np.ceil(len(split_set) / 2)), 2, ax + 1))

        axes[ax].set(title=f'{split_set[ax]} Images', xlabel=x_axis, ylabel=y_axis)
        
        axes[ax].set_ylim(ymin=0, ymax=1) 
        
        plots = []
        for i in sorted(list(set([split[2] for split in y_split]))):
            try:
                index = [(split[1], split[2]) for split in y_split].index(
                    (sorted(list(set([split[1] for split in y_split])))[ax], i)
                )
                plot = axes[ax].plot(
                    range(len(y_split[index][0])),
                    y_split[index][0],
                    color=cmap(y_split[index][2]),
                    label=str(y_split[index][2])
                )
                plots.append(plot)

                print(y_split[index][1],"|", y_split[index][2],"|", np.std(y_split[index][0]))
            except OSError:
                continue

        axes[ax].legend(loc='upper right')  

    fig.tight_layout()
    return fig
   
########## ########## ##########

if __name__ == "__main__":

    DATABASE_PATH = './data/full-scale.db'

    relative_cmap = True # if true: cmap values range from data min to data max || otherwise: cmap values range from 0 - 1

    LOG_TICKS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    f1 = plt.figure(1)
    f1 = heatmap(f1, relative_cmap=relative_cmap, log_scale=True, mode=1)    

    # f2 = plt.figure(2)
    # f2 = trial_history_graph(f2)

    # f3 = plt.figure(3)
    # f3 = trial_variability_comparison(f3)

    # f4 = plt.figure(4)
    # f4 = compute_time_graph(f4)

    # f5 = plt.figure(5)
    # f5 = deviation_graph(f5)

    plt.show()