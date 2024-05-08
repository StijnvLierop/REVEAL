import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
import argparse


def main(results_file: str,
         output_dir: str):
    """
    This function calculates metrics and creates a confusion matrix for
    a given evaluation result.

    :param results_file: A path to a .csv file containing the evaluation
    results for a particular dataset-detector combination. The file should
    have at least the following columns:
    - image (str): the name of the image that was evaluated.
    - true_value (bool): if the image is stego or not.
    - prediction (bool): if the image was predicted as stego or not.
    :param output_dir: A path to a directory where the output plots should be
    stored.
    """

    # Get results for this particular detector/dataset combination
    results = pd.read_csv(results_file)

    # Analyzed files
    print("Nr of files analyzed:", len(results))

    # Plot Confusion Matrix
    cm = confusion_matrix(results['true_value'],
                          results['prediction'],
                          labels=[0, 1])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['no stego', 'stego'])
    ax.yaxis.set_ticklabels(['no stego', 'stego'])
    plt.savefig(output_dir + "/confusion_matrix.png", format='png')

    # Metrics
    print("Accuracy:", accuracy_score(results['true_value'],
                                      results['prediction']))
    print("Precision:", precision_score(results['true_value'],
                                        results['prediction']))
    print("Recall:", recall_score(results['true_value'],
                                  results['prediction']))
    print("F1:", f1_score(results['true_value'],
                          results['prediction']))
    print("Balanced Accuracy:",
          balanced_accuracy_score(results['true_value'],
                                  results['prediction'])
          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--results-file',
        help='The file containing the structured results from the evaluation.',
        required=True,
        type=str
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        help='The location to export the visualizations.',
        required=True,
        type=str
    )
    args = parser.parse_args()
    main(**vars(args))