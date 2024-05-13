import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def main(dataset_master: str, tool_info: str):
    # Set font Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Read data
    tool_info = pd.read_excel(tool_info, decimal=',', index_col='ID')
    df = pd.read_csv(dataset_master, decimal=',', sep=';')

    # Only keep pictures that are actually present
    df = df.dropna(subset=['stegoPictureHash'])

    # Tool numbers to tool names
    df['toolname'] = df.apply(
        lambda x: tool_info.loc[int(x['tool'])]['Tool Name Short'], axis=1)

    # Determine images for which dimensions have been preserved
    df['filesizeChangeRelative'] = ((df['stegoPictureFilesize'] - df[
        'modifiedPictureFilesize']) / df['modifiedPictureFilesize']) * 100

    # Get average values
    sorted_by_change = df.groupby('toolname').mean(
        numeric_only=True).sort_values(by='filesizeChangeRelative').index

    # Plot
    fig = plt.figure(figsize=(15, 12))
    sns.barplot(df, x='toolname', y='filesizeChangeRelative',
                order=sorted_by_change, zorder=3)
    plt.xticks(rotation=0)
    plt.ylabel("Relative average change in file size (%)", size=25)
    plt.xlabel("Tool", size=25)
    plt.title("Relative Average Change in File Size per Tool", size=30, pad=20)
    fig.axes[0].spines['top'].set_visible(False)
    fig.axes[0].spines['right'].set_visible(False)
    fig.axes[0].grid(axis='y', zorder=0)
    plt.xticks(size=15, rotation=90)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig("change_in_filesize.png", dpi=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset-master',
        help='A path to the dataset master file (.csv).',
        required=True,
        type=str
    )
    parser.add_argument(
        '-t',
        '--tool-info',
        help='A path to a .xlsx file containing information on the tools that'
             ' were used to hide text in the pictures.',
        required=True,
        type=str
    )
    args = parser.parse_args()
    main(**vars(args))
