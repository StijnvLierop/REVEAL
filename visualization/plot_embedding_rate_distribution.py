import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns


def main(dataset_master: str):
    # Set font Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Read data
    df = pd.read_csv(dataset_master)

    # Exclude 0 ER
    df = df[df['embeddingRate'] != 0]

    # Make barplot
    plt.figure(figsize=(6, 4))
    sns.countplot(df, x='embeddingRate')

    plt.xlabel('Embedding rate (bpp)', fontsize=12)
    plt.ylabel('Nr of pictures', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("embedding_rate_distribution.svg", dpi=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset-master',
        help='A path to the dataset master file (.csv).',
        required=True,
        type=str
    )
    args = parser.parse_args()
    main(**vars(args))