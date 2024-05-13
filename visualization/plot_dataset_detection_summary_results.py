import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def main(detection_results: str):
    # Set font Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Read csv
    df = pd.read_excel(detection_results, decimal=',')

    # Remove Algorithms
    df = df[~df['Detector'].isin(['RS', 'SRNet', 'StegExpose'])]

    df['FPR'] = df['False Positives'] / (df['False Positives'] + df['True Negatives'])
    df['TPR'] = df['True Positives'] / (df['True Positives'] + df['False Negatives'])
    df['TNR'] = df['True Negatives'] / (df['True Negatives'] + df['False Positives'])
    df['FNR'] = df['False Negatives'] / (df['False Negatives'] + df['True Positives'])

    markers = ['o', '*', 's', 'd']
    colors = ['blue', 'red', 'orange', 'green'] # 4 tinten blauw
    datasets = df['Dataset'].unique()
    detectors = df['Detector'].unique()

    plt.figure(figsize=(20, 5))
    counter = 0
    for detector in detectors:
        df_detector = df[df['Detector'] == detector]
        for dataset, marker, color in zip(datasets, markers, colors):
            df_dataset_detector = df_detector[df_detector['Dataset'] == dataset]
            if 9 <= counter < 14:
                sns.scatterplot(df_dataset_detector, x=counter, y='FPR', marker=marker, s=70, facecolors='none', edgecolors=color, label=dataset+" FPR")
            else:
                sns.scatterplot(df_dataset_detector, x=counter, y='FPR', marker=marker, s=70, facecolors='none', edgecolors=color)
            counter+=1
        counter+=1

    counter = 0
    for detector in detectors:
        df_detector = df[df['Detector'] == detector]
        for dataset, marker, color in zip(datasets, markers, colors):
            df_dataset_detector = df_detector[df_detector['Dataset'] == dataset]
            if 9 <= counter < 14:
                sns.scatterplot(df_dataset_detector, x=counter, y='FNR', marker=marker, s=70, facecolors=color, edgecolors=color, label=dataset+" FNR")
            else:
                sns.scatterplot(df_dataset_detector, x=counter, y='FNR', marker=marker, s=70, facecolors=color, edgecolors=color)
            counter+=1
        counter += 1

    x_points = [4, 9, 14, 19, 24, 29, 34, 39]
    xticks = []
    xticks.append(x_points[0] / 2)
    for i in range(len(x_points) - 1):
        xticks.append((x_points[i] + x_points[i+1]) / 2)

    for x in x_points:
        plt.plot([x, x], [0, 1], dashes=[6, 2], color='black')

    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(detectors)

    plt.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
    plt.ylabel("Rate")
    plt.title("False Positive and False Negative Rates for different "
              "Dataset-Detector Combinations", size=15)
    plt.subplots_adjust(right=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("dataset_detection_results.png", dpi=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--detection-results',
        help='A path to the .xlsx file containing the detection results.',
        required=True,
        type=str
    )
    args = parser.parse_args()
    main(**vars(args))
