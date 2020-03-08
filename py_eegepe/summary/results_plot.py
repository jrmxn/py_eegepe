import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

sns.set()


def box_whisker(ax, df, cmap=None, labels=None):
    if not labels:
        labels = list(df.columns)
    ax.set_ylim([0.29, 2.01])

    bp = ax.boxplot(df.T, labels=labels, patch_artist=True, widths=0.5, showfliers=False)
    ax.set_xticks(range(1, 1 + df.shape[1]))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    arch_local = df.columns.to_list()
    if cmap is None:
        raise Exception('write this')

    for ix_arch, patch in enumerate(bp['boxes']):
        c = cmap[ix_arch]
        bp['boxes'][ix_arch].set_facecolor([1, 1, 1])
        bp['boxes'][ix_arch].set_edgecolor(c)
        bp['medians'][ix_arch].set_color(c)
        bp['whiskers'][ix_arch * 2 + 0].set_color(c)
        bp['whiskers'][ix_arch * 2 + 1].set_color(c)
        bp['caps'][ix_arch * 2 + 0].set_color(c)
        bp['caps'][ix_arch * 2 + 1].set_color(c)

        bp['boxes'][ix_arch].set_linewidth(1.5)
        bp['caps'][ix_arch * 2 + 0].set_linewidth(1)
        bp['caps'][ix_arch * 2 + 1].set_linewidth(1)
        bp['medians'][ix_arch].set_linewidth(1)
        bp['whiskers'][ix_arch * 2 + 0].set_linewidth(1)
        bp['whiskers'][ix_arch * 2 + 1].set_linewidth(1)
        bp['whiskers'][ix_arch * 2 + 0].set_linestyle('--')
        bp['whiskers'][ix_arch * 2 + 1].set_linestyle('--')

    c_lines = 0.7 * np.ones(3)
    for ix_sub in range(df.shape[0]):
        for ix_pair in range(len(arch_local) - 1):
            x1 = df.iloc[ix_sub].loc[arch_local[ix_pair + 0]]
            x2 = df.iloc[ix_sub].loc[arch_local[ix_pair + 1]]
            ax.plot([ix_pair + 1, ix_pair + 2], [x1, x2], 'o-', color=c_lines, markersize=3, markerfacecolor=c_lines,
                     markeredgewidth=0.5, markeredgecolor=[1, 1, 1], linewidth=0.5, zorder=3, alpha=0.95)

    p, pairs = get_paired_p_values(df, labels=labels)
    y_step = 0.05
    y_upper_bound = ax.get_ylim()[1] - y_step  # -(1+len(pairs)) * y_step
    add_stars(ax, pairs, p, y_upper_bound, y_step=y_step, bar_edge=0.1)
    ax.set_ylabel('MACE (rad.)')
    ax.plot(ax.get_xlim(), np.pi / 2 * np.ones(2), '--', color=0.75*np.ones(3))

    # just a manual check...
    # multipletests([stats.wilcoxon(df.loc[:, 'test_net_cug_000'], df.loc[:, 'test_fir_gen_000'])[1],
    #                stats.wilcoxon(df.loc[:, 'test_net_cug_000'], df.loc[:, 'test_fir_alp_000'])[1],
    #                stats.wilcoxon(df.loc[:, 'test_net_cug_000'], df.loc[:, 'test_net_sgd_000'])[1],
    #                stats.wilcoxon(df.loc[:, 'test_net_cug_000'], df.loc[:, 'test_net_sgd_001'])[1]])

    return p, pairs


def get_paired_p_values(df, labels=None):
    arch_local = df.columns.to_list()
    pairs = list(combinations(range(len(arch_local)), 2))
    if not labels:
        labels = arch_local
    p = []
    md = []
    lower = []
    for comb in pairs:
        x = df.loc[:, arch_local[comb[0]]]
        y = df.loc[:, arch_local[comb[1]]]
        md.append(np.mean(x-y))
        lower.append(np.mean(x - y) < 0)
        st = stats.wilcoxon(x, y)
        p.append(st.pvalue)
        # md.append() # code for median difference

    method = 'bonferroni'
    print(f'Doing a {method} correction.')
    p_corr = multipletests(np.array(p), method=method)
    p_corr = p_corr[1]

    for ix_pairs in range(len(pairs)):
        lower_str = arch_local[comb[0]] if lower[ix_pairs] else arch_local[comb[1]]
        print(f'{labels[pairs[ix_pairs][0]]} vs {labels[pairs[ix_pairs][1]]}' +
              f', MD = {md[ix_pairs]:0.2f}, p = {p_corr[ix_pairs]:0.4f} [Lower: {lower_str}]')

    from tabulate import tabulate
    table_p = np.zeros((1+np.max(pairs), 1+np.max(pairs))) * np.nan
    table_md = np.zeros((1+np.max(pairs), 1+np.max(pairs))) * np.nan
    for ix_pairs in range(len(pairs)):
        table_p[pairs[ix_pairs][0], pairs[ix_pairs][1]] = p_corr[ix_pairs]
        table_md[pairs[ix_pairs][0], pairs[ix_pairs][1]] = md[ix_pairs]
    tablefmt = 'plain'
    # tablefmt = 'latex'
    print('p-values')
    print(tabulate(table_p, headers=labels, tablefmt=tablefmt, floatfmt=".4f"))
    print('MD (median differences)')
    print(tabulate(table_md, headers=labels, tablefmt=tablefmt, floatfmt=".3f"))


    return np.array(p_corr), np.array(pairs)


def add_stars(ax, pairs, p, y0, y_step=0.015, bar_edge=0.1):
    diff_pairs = np.diff(pairs, axis=1).squeeze()
    ix_sort = np.argsort(diff_pairs)[::-1]
    p = p[ix_sort]
    pairs = pairs[ix_sort]
    diff_pairs = diff_pairs[ix_sort]
    ix_valid = 0
    for ix, pair in enumerate(pairs):
        if p[ix] < 0.05:
            y = y0 - ix_valid * np.ones(2) * y_step
            n_stars = int(p[ix] < 0.05) + int(p[ix] < 0.01) + int(p[ix] < 0.001)
            dm = 0.1 + 0.05 * n_stars
            zorder = 4
            c_gray = np.ones(3) * 0.35
            x = np.array([pair[0] + bar_edge + 1, pair[1] - bar_edge + 1])
            x1 = np.array([x[0], np.mean(x) - dm])
            x2 = np.mean(x)
            x3 = np.array([np.mean(x) + dm, x[1]])
            ax.plot(x1, y, 'k-', zorder=zorder, color=c_gray)
            ax.text(x2, y[0]-0.025, '*'*n_stars, horizontalalignment='center',
                    verticalalignment='center_baseline', color=c_gray, zorder=zorder)
            ax.plot(x3, y, '-', zorder=zorder, color=c_gray)

            if diff_pairs[ix] > 1:
                # keep the bars in one line if they are local paired differences (in the plot)
                ix_valid = ix_valid + 1



if __name__ == '__main__':
    """
    Example usage - 
    """
    from pathlib import Path
    import pandas as pd
    f_width = 4
    f_height = 6
    plt.close('all')
    ix_fig = 99
    img_fmt = 'png'
    dpi = 150
    dpi_print = 300
    figures = Path.cwd()

    fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)

    df = pd.DataFrame(np.random.randint(0, 15, size=(15, 4)), columns=list('ABCD'))
    df.rename(index={ix:f"sub{ix}" for ix in df.index}, inplace=True)
    cmap = sns.color_palette("colorblind", len(df.columns))

    box_whisker(ax, df, cmap)
    ax.set_ylabel('something')
    fig.tight_layout()

    f_name = f"paper_{''}_fig_{ix_fig}e.{img_fmt}"
    plt.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)