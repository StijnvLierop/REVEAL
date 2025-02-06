from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

# Category mapping
CATEGORY_MAPPING = {
    "no modifications" : "no preprocessing operations",
    "to JPEG (conversion) PIL DCT quality preset maximum" : "filetype conversion",
    "to JPEG (conversion) PIL DCT quality preset medium" : "filetype conversion",
    "to JPEG (conversion) PIL DCT quality preset low" : "filetype conversion",
    "to JPEG (conversion) PIL DCT quality preset web_high" : "filetype conversion",
    "to JPEG (conversion) PIL DCT quality preset web_low" : "filetype conversion",
    "make file type compatible (to gif)" : "filetype conversion",
    "make file type compatible (to jpeg)" : "filetype conversion",
    "make file type compatible (to jpg)" : "filetype conversion",
    "make file type compatible (to png)" : "filetype conversion",
    "make file type compatible (to bmp)" : "filetype conversion",
    "to PNG" : "filetype conversion",
    "to GIF" : "filetype conversion",
    "to BMP" : "filetype conversion",
    "upsampling to 1920x1080" : "resolution change",
    "downsampling to 1920x1080" :  "resolution change",
    "downsampling to 512x512":  "resolution change",
    "downsampling to 256x256":  "resolution change",
    "upsampling to 1280x720" :  "resolution change",
    "downsampling to 1280x720":  "resolution change",
    "upsampling to 1080x1080" :  "resolution change",
    "downsampling to 1080x1080":  "resolution change",
    "upsampling to 3840x2160" :  "resolution change",
    "downsampling to 3840x2160" : "resolution change",
    "upsampling to 4096x2160" : "resolution change",
    "downsampling to 4096x2160" : "resolution change",
    "upsampling to 500x500" :  "resolution change",
    "downsampling to 500x500" :  "resolution change",
    "upsampling to 7680x4320" :  "resolution change",
    "crop to 1:1" : "crop",
    "crop to 3:2" : "crop",
    "crop to 5:4" : "crop",
    "crop to 16:9" : "crop",
    "crop to 9:16" : "crop",
    "log" : "compression",
    "to JPEG (compression) PIL DCT quality preset maximum" : "compression",
    "to JPEG (compression) PIL DCT quality preset medium" : "compression",
    "to JPEG (compression) PIL DCT quality preset low": "compression",
    "to JPEG (compression) PIL DCT quality preset web_high": "compression",
    "to JPEG (compression) PIL DCT quality preset web_low": "compression",
    "to RGB": "colour model change",
    "to CMYK and back to RGB": "colour model change",
    "to RGBA (and PNG)": "colour model change",
    "to L": "colour model change",
    "to HSV and back to RGB": "colour model change",
    "zip and unzip" : "compression",
    "EXIF wipe" : "other",
    "mirror" : "other",
    "rotate 90" : "rotate",
    "rotate -90" :  "rotate",
    "rotate 180" :  "rotate",
    "screenshot and crop" : "other"
}

# Define a custom formatter
def dot_thousands(x, pos):
    return f"{x:,.0f}".replace(",", ".")

def main(dataset_master: str):
    # Set font Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Read data
    df = pd.read_csv(dataset_master)

    # Count occurrences of all preprocessing operations
    counts = df.apply(lambda row: Counter(row['modifications'].split(' - ')), axis=1).sum()

    # Group preprocessing operations in categories
    category_counts = {}
    for item, count in counts.items():
        # Find the category for the given item
        category = CATEGORY_MAPPING.get(item)

        # Add the count to the corresponding category in category_counts
        if category:
            if category in category_counts:
                category_counts[category] += count
            else:
                category_counts[category] = count
    category_counts = pd.DataFrame.from_dict(category_counts, orient='index').reset_index()
    category_counts.rename(columns={'index': 'category'}, inplace=True)

    # Make barplot
    plt.figure(figsize=(6, 4))
    plt.grid(axis='x', zorder=0)
    sns.barplot(category_counts, y='category', x=0, zorder=3)

    plt.ylabel('Preprocessing operation category', fontsize=12)
    plt.xlabel('Nr of times applied', fontsize=12)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(dot_thousands))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("modifications_per_general_category.svg", dpi=600)


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