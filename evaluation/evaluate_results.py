import pandas as pd
from sklearn.metrics import *
import argparse


def main(results_file: str,
         output_dir: str):

    # Get results for this particular detector/dataset combination
    results = pd.read_csv(results_file, sep=';')

    # Analyzed files
    print("Nr of files analyzed:", len(results))

    # Plot Confusion Matrix
    cm = confusion_matrix(results['true_value'],
                          results['prediction'],
                          labels=[0, 1])
    cmd = ConfusionMatrixDisplay(cm)
    cmd.figure_.savefig(output_dir + "/confusion_matrix.png", format='png')

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
        help='The file containing the raw results from the evaluation.',
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