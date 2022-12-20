import sklearn 
from matplotlib import pyplot as plt

def plot_calibration_curve(y_true, y_prob, plot_name):
    fraction_of_positives, mean_predicted_value = sklearn.calibration.calibration_curve(y_true, y_prob, n_bins=5)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Calibration plot (reliability curve)")
    plt.legend()
    plt.show()
    plt.savefig(plot_name)
    