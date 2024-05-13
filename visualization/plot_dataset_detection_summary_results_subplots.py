import pandas as pd
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

    markers = ['o', '*', 's', 'd', 'v', 'D', 'P', '>']
    colors = ['blue', 'red', 'orange', 'green']
    datasets = df['Dataset'].unique()
    detectors = df['Detector'].unique()

    fig = plt.figure(figsize=(10, 6))

    for d, detector in enumerate(detectors):
        data = df[df['Detector'] == detector]
        for k, dataset in enumerate(datasets):
            data_sub = data[data['Dataset'] == dataset]
            plt.scatter(data_sub['FNR'], data_sub['FPR'], color=colors[k], marker=markers[d])

    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)

    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in range(len(datasets))]
    handles += [f(markers[i], "k") for i in range(len(detectors))]

    plt.subplots_adjust(bottom=0.2, hspace=0.6, wspace=0.4)
    plt.figlegend(handles=handles, labels=list(datasets)+list(detectors), loc='lower center',  ncol=4)
    plt.suptitle("False Positive and False Negative Rates for different Dataset-Detector Combinations", size=15)
    plt.savefig('detection_performance_new.png', dpi=300)
    plt.savefig("dataset_detection_results_subplots.png", dpi=600)

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

