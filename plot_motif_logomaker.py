import logomaker
from matplotlib import pyplot
from matplotlib.pyplot import MultipleLocator

def plot_motif_logomaker(cluster_name, seqs):
    ww_counts_df = logomaker.alignment_to_matrix(sequences=seqs, to_type='counts', characters_to_ignore='.-X')
    figsize=(4,3)
    figure1, ax1 = pyplot.subplots(figsize=figsize)
    ww_info_df = logomaker.transform_matrix(ww_counts_df, 
                                            from_type='counts', 
                                            to_type='information')

    logomaker.Logo(ww_info_df, figsize=figsize, color_scheme='dmslogo_funcgroup', ax=ax1)
    ax1.set_yticks([])
    x_major_locator=MultipleLocator(1)
    ax1.xaxis.set_major_locator(x_major_locator)

    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_xlabel('Position',fontsize=10)
    ax1.set_xticks(range(9))
    ax1.set_xticklabels('%d'%x for x in list(range(1,9+1)))
    figure1.savefig(cluster_name, dpi=300, bbox_inches = 'tight')