import matplotlib.pyplot as plt
import pandas as pd
import argparse


def main(dataset_master: str, camera_info: str):
    # Set font Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Read data
    df = pd.read_csv(dataset_master, decimal=',', sep=';')
    camera_df = pd.read_csv(camera_info, sep=';')

    # Add camera information to dataframe
    camera_df['camera'] = (camera_df['Make'] + ' ' + camera_df['Model']).astype(str)
    camera_df['camera'] = camera_df['camera'].str.replace('Apple ', '')
    camera_df['camera'] = camera_df['camera'].str.replace('Stable Diffusion ', '')
    camera_df = camera_df[['camera', 'Make', 'Model']]
    df['camera'] = df['camera'].astype(str)
    df = pd.merge(df, camera_df, on='camera', how='left')

    # Count cameras
    df['counts'] = 1
    df_grouped = df.groupby(['Make', 'Model'])['counts'].sum().reset_index()

    # Make barplot
    plt.figure(figsize=(6, 10))
    makes = []
    for i, make in enumerate(df['Make'].unique()):
        make_models = df_grouped[df_grouped['Make'] == make]
        makes.append(make)
        plt.barh(make_models['Model'], make_models['counts'], label=make, zorder=3)

    # Plot lines
    plt.text(0.05, 0.18, makes[0], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.25, 0.25], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.29, makes[1], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.282, 0.282], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.435, makes[2], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.571, 0.571], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.605, makes[3], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.667, 0.667], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.67, makes[4], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.715, 0.715], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.72, makes[5], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.78, 0.78], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.77, makes[6], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.825, 0.825], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.815, makes[7], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.875, 0.875], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.858, makes[8], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.923, 0.923], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.885, makes[9], fontsize=12, transform=plt.gcf().transFigure)
    line = plt.Line2D([-2, 1], [0.94, 0.94], color='black', transform=plt.gca().transAxes)
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.text(0.05, 0.901, makes[10], fontsize=12, transform=plt.gcf().transFigure)

    plt.xlabel('Nr of pictures')
    plt.gca().xaxis.grid(True, zorder=0)
    plt.subplots_adjust(left=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("camera_distribution.png", dpi=600)


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
        '-c',
        '--camera-info',
        help='A path to a .csv file containing information on the cameras that'
             ' were used to take the pictures.',
        required=True,
        type=str
    )
    args = parser.parse_args()
    main(**vars(args))