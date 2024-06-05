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

    # Remove our dataset from plot
    df_filtered = df[df['Dataset'] != "Ours"]

    # Get unique datasets and detectors
    datasets = df['Dataset'].unique()
    detectors = df['Detector'].unique()

    # Create figure
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))

    for d, ax in enumerate(axs.ravel()):
        data = df_filtered[df_filtered['Detector'] == detectors[d]]
        for k, dataset in enumerate(datasets):
            data_sub = data[data['Dataset'] == dataset]
            ax.scatter(data_sub['FNR_diff'], data_sub['FPR_diff'], label=dataset)
            ax.set_xlabel('False Negative Rate')
            ax.set_ylabel('False Positive Rate')
            ax.set_ylim(-0.76, 0.76)
            ax.set_xlim(-0.3, 0.3)
            ax.hlines(y=0, xmin=-0.3, xmax=0.3, color='gray', linestyle='--', zorder=0)
            ax.vlines(x=0, ymin=-0.75, ymax=0.75, color='gray', linestyle='--', zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            FPR_Ours = round(df.loc[(df['Detector'] == detectors[d]) & (df['Dataset'] == 'Ours'), 'FPR'].to_list()[0], 1)
            FNR_Ours = round(df.loc[(df['Detector'] == detectors[d]) & (df['Dataset'] == 'Ours'), 'FNR'].to_list()[0], 1)
            ax.set_title(f"{detectors[d]}\n")
            ax.text(0, 0.85, f'(Ours: FPR={FPR_Ours}, FNR={FNR_Ours})', fontsize=9, ha='center')

    plt.subplots_adjust(bottom=0.2, hspace=0.6, wspace=0.4)
    plt.legend(loc='lower center', bbox_to_anchor=(1.3, 1))
    plt.suptitle("Difference in False Positive and False Negative Rates for different Dataset-Detector Combinations", size=15)
    plt.savefig("dataset_detection_results_relative_subplots.png", dpi=600)


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
