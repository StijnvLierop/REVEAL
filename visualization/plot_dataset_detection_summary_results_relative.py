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

    # Calculate rates
    df['FPR'] = df['False Positives'] / (df['False Positives'] + df['True Negatives'])
    df['TPR'] = df['True Positives'] / (df['True Positives'] + df['False Negatives'])
    df['TNR'] = df['True Negatives'] / (df['True Negatives'] + df['False Positives'])
    df['FNR'] = df['False Negatives'] / (df['False Negatives'] + df['True Positives'])

    # Calculate difference in rates between our dataset and other datasets
    for key, row in df.iterrows():
        # Get detector
        detector = row['Detector']

        # Calculate differences
        df.loc[key, 'FPR_diff'] = df.loc[key, 'FPR'] - df.loc[(df['Detector'] == detector) & (df['Dataset'] == 'Ours'), 'FPR'].to_list()[0]
        df.loc[key, 'TPR_diff'] = df.loc[key, 'TPR'] - df.loc[(df['Detector'] == detector) & (df['Dataset'] == 'Ours'), 'TPR'].to_list()[0]
        df.loc[key, 'TNR_diff'] = df.loc[key, 'TNR'] - df.loc[(df['Detector'] == detector) & (df['Dataset'] == 'Ours'), 'TNR'].to_list()[0]
        df.loc[key, 'FNR_diff'] = df.loc[key, 'FNR'] - df.loc[(df['Detector'] == detector) & (df['Dataset'] == 'Ours'), 'FNR'].to_list()[0]

    # Set markers and colors
    markers = ['o', '*', 's', 'd', 'v', 'D', 'P', '>']
    colors = ['blue', 'red', 'green']

    # Remove our dataset from plot
    df = df[df['Dataset'] != "Ours"]

    # Get unique datasets and detectors
    datasets = df['Dataset'].unique()
    detectors = df['Detector'].unique()

    # Create figure
    plt.figure(figsize=(10, 6))

    for d, detector in enumerate(detectors):
        data = df[df['Detector'] == detector]
        for k, dataset in enumerate(datasets):
            data_sub = data[data['Dataset'] == dataset]
            plt.scatter(data_sub['FNR_diff'], data_sub['FPR_diff'], color=colors[k], marker=markers[d])

    # Labels and formatting
    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.ylim(-0.76, 0.76)
    plt.xlim(-0.3, 0.3)
    plt.hlines(y=0, xmin=-0.3, xmax=0.3, color='gray', linestyle='--', zorder=0)
    plt.vlines(x=0, ymin=-0.75, ymax=0.75, color='gray', linestyle='--', zorder=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in range(len(datasets))]
    handles += [f(markers[i], "k") for i in range(len(detectors))]

    plt.subplots_adjust(bottom=0.2, hspace=0.6, wspace=0.4)
    plt.figlegend(handles=handles, labels=list(datasets)+list(detectors), loc='lower center',  ncol=4)
    plt.suptitle("Difference in False Positive and False Negative Rates for different Dataset-Detector Combinations", size=15)
    plt.savefig("dataset_detection_results_relative.png", dpi=600)


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
