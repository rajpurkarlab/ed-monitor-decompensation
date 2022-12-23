from sklearn import calibration
from matplotlib import pyplot as plt
import numpy as np

def plot_calibration_curve(y_true, y_prob, plot_name):
    # sort y_prob from smallest to largest and sort y_true with the same indexing
    y_prob, y_true = zip(*sorted(zip(y_prob, y_true)))

    # divide y_prob and y_true into 5 sublists
    y_prob = np.array_split(y_prob, 5)
    y_true = np.array_split(y_true, 5)

    # calculate the mean of each sublist in y_prob and store in a list
    mean_predicted_value = []
    for i in range(len(y_prob)):
        mean_predicted_value.append(np.mean(y_prob[i]))

    # calculate the fraction of positives in each sublist in y_true and store in a list
    fraction_of_positives = []
    for i in range(len(y_true)):
        fraction_of_positives.append(np.mean(y_true[i]))
            
    min_x = 0.9 * min(min(mean_predicted_value), min(fraction_of_positives))
    max_y = 1.1 * max(max(mean_predicted_value), max(fraction_of_positives))
    
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([min_x, max_y], [min_x, max_y], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Calibration plot (reliability curve)")
    plt.legend()
    plt.show()
    plt.savefig(plot_name)
    
