import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


def parse_arguments():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--foldername', type=str, default='data/output/')
    parser.add_argument('--figtype', type=str, default='.png')
    parser.add_argument('--ma_window', type=int, default=2)
    return parser.parse_args()


def extract_subfolder(foldername):
    listfolder = [f.path for f in os.scandir(foldername) if f.is_dir()]
    return listfolder


def compare_curve(listdir, window):
    df = pd.DataFrame([])
    for subfolder in listdir:
        df_sub = pd.read_csv(subfolder+'/test_result.txt', sep=',', skipinitialspace=True)
        df_sub['reward_mean'] = df_sub['reward'].rolling(window).mean()
        df_sub['collision_mean'] = df_sub['collision'].rolling(window).mean()
        df_sub['reward_std'] = df_sub['reward'].rolling(window).std()
        df_sub['collision_std'] = df_sub['collision'].rolling(window).std()
        name = subfolder.split('-')[1]
        if name == 'event': name = 'ours'
        df_sub['method'] = name
        df = df.append(df_sub)

    sns.set_style("darkgrid")

    # fig 1
    reward_plot = sns.lineplot(
        data=df,
        x="epoch", y="reward_mean", hue="method",
        markers=False, dashes=False, palette="flare"
    )
    reward_plot.set(xlabel='Epoch', ylabel='Reward')
    reward_plot.legend_.set_title(None)

    figname = args.foldername + 'reward' + args.figtype
    fig = reward_plot.get_figure()
    fig.set_size_inches(4, 3)
    fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.clear()

    # fig 2
    df['collision_mean'] *= 100
    col_plot = sns.lineplot(
        data=df,
        x="epoch", y="collision_mean", hue="method",
        markers=False, dashes=False, palette="flare"
    )
    col_plot.set(xlabel='Epoch', ylabel='Collision (%)')
    col_plot.set(ylim=(0.0, 55.0), xlabel='Epoch', ylabel='Collision (%)')
    col_plot.legend_.set_title(None)

    figname = args.foldername + 'collision' + args.figtype
    fig = col_plot.get_figure()
    fig.set_size_inches(4, 3)
    fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.clear()


if __name__ == "__main__":
    args = parse_arguments()
    subfolders = extract_subfolder(args.foldername)
    compare_curve(subfolders, args.ma_window)
