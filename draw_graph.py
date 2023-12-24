import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt
from config import root_path


def main():
    model_name = 'xjy_20231218'
    performance_path = os.path.join(root_path, 'datasets', model_name, 'performance.csv')
    performance = pd.read_csv(performance_path)

    plt.figure()
    plt.show()


if __name__ == '__main__':
    main()
