import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns


def main(complexity_color_image_datasets: str):
    # Set font Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Read data
    df = pd.read_csv(complexity_color_image_datasets)

    # Plot Complexity and Color distribution for every dataset in one plot
    fig = plt.figure(figsize=(6, 5))
    sns.boxplot(df, x='Complexity', y='Dataset')
    plt.title('Complexity Distribution of various Image Datasets', fontsize=15, pad=20)
    plt.xlabel('Complexity')
    plt.ylabel('Dataset')
    fig.axes[0].spines['top'].set_visible(False)
    fig.axes[0].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('CompComparison.svg', dpi=600)

    fig = plt.figure(figsize=(6, 5))
    sns.boxplot(df, x='AvgHue', y='Dataset')
    plt.title('Color Distribution of various Image Datasets', fontsize=15, pad=20)
    plt.xlabel('Average Hue')
    plt.ylabel('Dataset')
    fig.axes[0].spines['top'].set_visible(False)
    fig.axes[0].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('ColorComparison.svg', dpi=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--complexity_color_image_datasets',
        help='A path to the file (.csv) with complexity and color calculated for various datasets.',
        required=True,
        type=str
    )
    args = parser.parse_args()
    main(**vars(args))