import os

import numpy as np
from matplotlib import pyplot as plt


def bar_chart(values, tick_labels, x_label, y_label, title="", output_path=os.path.join('data', 'plots', 'bar_chart.png')):
    index = np.arange(len(tick_labels))
    plt.bar(index, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(index, tick_labels)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()