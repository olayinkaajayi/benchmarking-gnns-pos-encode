import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



def read_csv(filename, matrix=False):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)

        if not matrix:
            for row in reader:
                data.append(float(row[0]))
        else:
            for row in reader:
                data.append(np.float_(row))
            data = np.array(data)

    return data




def combine_plot_line_with_error(data1, data2, datasets, folders, savefilename, exp='eta'):
    
    x1 = [1e3, 3e3, 5e3, 7e3, 9e3, 1e4, 5e4, 8e4, 10e4, 11e4, 12e4, 5e5] #WikiCS
    # x1 = [1e4, 5e4, 7e4, 1e5, 5e5, 7e5] #WikiCS
    x2 = [1e3, 5e3, 1e4, 2e4, 5e4, 10e4, 15e4, 5e5] # Pubmed
    
    # WikiCS
    mean_test1 = data1[:,2]
    std_test1 = data1[:,3]

    #Pubmed
    mean_test2 = data2[:,2]
    std_test2 = data2[:,3]

    plt.figure()
    # WikiCS
    plt.plot(x1, mean_test1, marker='o', markersize=mk_size, color='g', label="WikiCS")
    plt.errorbar(x1, mean_test1, yerr = std_test1, fmt ='o', color='g', capsize=3)

    # Pubmed
    plt.plot(x2, mean_test2, marker='o', markersize=mk_size, color='brown', label="PubMed")
    plt.errorbar(x2, mean_test2, yerr = std_test2, fmt ='o', color='brown', capsize=3)

    # Rescale x-axis
    ax = plt.gca()
    ax.set_xscale('log')
    # plt.ylim(82, 96)
    plt.ylim(np.min(mean_test1)-np.max(std_test1)-1, np.max(mean_test2)+np.max(std_test2)+1)
    plt.title(f'Effect of $\eta$ on Accuracy for {datasets[0]} and {datasets[1]}')
    plt.xlabel(f'$\eta$')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.savefig(savefilename)


def combine_plot_line_with_error_same(data, datasets, folders, savefilename, exp='eta'):
    
    wiki = [1e3, 3e3, 5e3, 7e3, 9e3, 1e4, 5e4, 8e4, 10e4, 11e4, 12e4, 5e5] #WikiCS
    pub = [1e3, 5e3, 1e4, 2e4, 5e4, 10e4, 15e4, 5e5] # Pubmed

    data1_df_ns,data1_df_s,data1_bf_ns,data1_bf_s,data2_df_ns,data2_df_s,data2_bf_ns,data2_bf_s = data
    
    # WikiCS depth-first: no-sign
    mean_test1_df_ns = data1_df_ns[:,2]
    std_test1_df_ns = data1_df_ns[:,3]

    # WikiCS breadth-first: no-sign
    mean_test1_bf_ns = data1_bf_ns[:,2]
    std_test1_bf_ns = data1_bf_ns[:,3]

    # WikiCS depth-first: sign
    mean_test1_df_s = data1_df_s[:,2]
    std_test1_df_s = data1_df_s[:,3]

    # WikiCS breadth-first: sign
    mean_test1_bf_s = data1_bf_s[:,2]
    std_test1_bf_s = data1_bf_s[:,3]
    ########################################################
    # PubMed depth-first: no-sign
    mean_test2_df_ns = data2_df_ns[:,2]
    std_test2_df_ns = data2_df_ns[:,3]

    # PubMed breadth-first: no-sign
    mean_test2_bf_ns = data2_bf_ns[:,2]
    std_test2_bf_ns = data2_bf_ns[:,3]

    # PubMed depth-first: sign
    mean_test2_df_s = data2_df_s[:,2]
    std_test2_df_s = data2_df_s[:,3]

    # PubMed breadth-first: sign
    mean_test2_bf_s = data2_bf_s[:,2]
    std_test2_bf_s = data2_bf_s[:,3]

    # Create a figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2 , figsize=(15, 15))

    # Set the font size
    fontsize = 25
    fnt_tick = 20
    s_mark = 450
    mk_size = 15
    plt.rcParams['font.size'] = fontsize

    # WikiCS: no-sign, breadth-first
    ax1.plot(wiki, mean_test1_bf_ns, marker='o', markersize=mk_size, color='b', label="breadth-first")
    ax1.errorbar(wiki, mean_test1_bf_ns, yerr = std_test1_bf_ns, fmt ='o', color='b', capsize=3)
    # WikiCS: no-sign, depth-first
    ax1.plot(wiki, mean_test1_df_ns, marker='d', linestyle='dashed', color='r', label="depth-first")
    ax1.errorbar(wiki, mean_test1_df_ns, yerr = std_test1_df_ns, fmt ='^', markersize=mk_size, color='r', capsize=3)
    ax1.set_ylabel('Accuracy (%)',fontsize=fontsize)
    ax1.set_xlabel(f'$\eta$',fontsize=fontsize)

    # WikiCS: sign, breadth-first
    ax2.plot(wiki, mean_test1_bf_s, marker='o', markersize=mk_size, color='b', label="breadth-first")
    ax2.errorbar(wiki, mean_test1_bf_s, yerr = std_test1_bf_s, fmt ='o', color='b', capsize=3)
    # WikiCS: sign, depth-first
    ax2.plot(wiki, mean_test1_df_s, marker='d', linestyle='dashed', color='r', label="depth-first")
    ax2.errorbar(wiki, mean_test1_df_s, yerr = std_test1_df_s, fmt ='^', markersize=mk_size, color='r', capsize=3)
    ax2.set_ylabel('Accuracy (%)',fontsize=fontsize)
    ax2.set_xlabel(f'$\eta$',fontsize=fontsize)    

    # Pubmed: no-sign, breadth-first
    ax3.plot(pub, mean_test2_bf_ns, marker='o', markersize=mk_size, color='b', label="breadth-first")
    ax3.errorbar(pub, mean_test2_bf_ns, yerr = std_test2_bf_ns, fmt ='o', color='b', capsize=3)
    # Pubmed: no-sign, depth-first
    ax3.plot(pub, mean_test2_df_ns, marker='d', linestyle='dashed', color='r', label="depth-first")
    ax3.errorbar(pub, mean_test2_df_ns, yerr = std_test2_df_ns, fmt ='^', markersize=mk_size, color='r', capsize=3)
    ax3.set_ylabel('Accuracy (%)',fontsize=fontsize)
    ax3.set_xlabel(f'$\eta$',fontsize=fontsize)

    # Pubmed: sign, breadth-first
    ax4.plot(pub, mean_test2_bf_s, marker='o', markersize=mk_size, color='b', label="breadth-first")
    ax4.errorbar(pub, mean_test2_bf_s, yerr = std_test2_bf_s, fmt ='o', color='b', capsize=3)
    # Pubmed: sign, depth-first
    ax4.plot(pub, mean_test2_bf_s, marker='d', linestyle='dashed', color='r', label="depth-first")
    ax4.errorbar(pub, mean_test2_bf_s, yerr = std_test2_bf_s, fmt ='^', markersize=mk_size, color='r', capsize=3) 
    ax4.set_ylabel('Accuracy (%)',fontsize=fontsize)
    ax4.set_xlabel(f'$\eta$',fontsize=fontsize)

    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    ax1.set_title("WikiCS, NO sign flip")
    ax2.set_title("WikiCS, sign flip")
    ax3.set_title("PubMed, NO sign flip")
    ax4.set_title("PubMed, sign flip")

    ax1.tick_params(axis="x", labelsize=fnt_tick)
    ax2.tick_params(axis="x", labelsize=fnt_tick)
    ax3.tick_params(axis="x", labelsize=fnt_tick)
    ax4.tick_params(axis="x", labelsize=fnt_tick)

    ax1.tick_params(axis="y", labelsize=fnt_tick)
    ax2.tick_params(axis="y", labelsize=fnt_tick)
    ax3.tick_params(axis="y", labelsize=fnt_tick)
    ax4.tick_params(axis="y", labelsize=fnt_tick)
    

    wiki_m_ns = np.concatenate([mean_test1_bf_ns,mean_test1_bf_s])
    wiki_m_s = np.concatenate([mean_test1_bf_ns,mean_test1_bf_s])
    wiki_s_ns = np.concatenate([std_test1_bf_ns,std_test1_bf_s])
    wiki_s_s = np.concatenate([std_test1_bf_ns,std_test1_bf_s])

    pub_m_ns = np.concatenate([mean_test2_bf_ns,mean_test2_bf_s])
    pub_m_s = np.concatenate([mean_test2_bf_ns,mean_test2_bf_s])
    pub_s_ns = np.concatenate([std_test2_bf_ns,std_test2_bf_s])
    pub_s_s = np.concatenate([std_test2_bf_ns,std_test2_bf_s])

    ax1.set_ylim(np.min(wiki_m_ns)-np.max(wiki_s_ns)-1.0, np.max(wiki_m_ns)+np.max(wiki_s_ns)+1.0)
    ax2.set_ylim(np.min(wiki_m_s)-np.max(wiki_s_s)-1.0, np.max(wiki_m_s)+np.max(wiki_s_s)+1.0)
    ax3.set_ylim(np.min(pub_m_ns)-np.max(pub_s_ns)-0.1, np.max(pub_m_ns)+np.max(pub_s_ns)+0.1)
    ax4.set_ylim(np.min(pub_m_s)-np.max(pub_s_s)-0.1, np.max(pub_m_s)+np.max(pub_s_s)+0.1)

    fig.suptitle(f'Effect of $\eta$ on Accuracy for {datasets[0]} and {datasets[1]} datasets')

    # set legend position
    # lines = [] 
    # labels = [] 
    # for i,ax in enumerate(fig.axes):
    #     if (i+1) >= 2:
    #         break
    #     Line, Label = ax.get_legend_handles_labels()
    #     lines.extend(Line) 
    #     labels.extend(Label)
    
    # fig.legend(labels, loc='lower center') 
    # fig.legend(bbox_to_anchor=(1.3, 0.6), loc='lower center')

    # Add inset plot to ax2
    x_min, x_max, y_min, y_max = 0.5e5, 2e5, 77.65, 77.75
    axins = inset_axes(ax2, width="30%", height="30%", loc='lower left', borderpad=2)
    axins.plot(wiki, mean_test1_bf_s, marker='o', markersize=mk_size, color='b')
    axins.errorbar(wiki, mean_test1_bf_s, yerr = std_test1_bf_s, fmt ='o', color='b', capsize=3)
    axins.plot(wiki, mean_test1_df_s, marker='d', linestyle='dashed', color='r')
    axins.errorbar(wiki, mean_test1_df_s, yerr = std_test1_df_s, fmt ='^', markersize=mk_size, color='r', capsize=3)
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.set_xscale('log')

    # Hide x and y ticks for the inset plot
    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Adjust layout
    plt.tight_layout()

    # save the final image
    plt.savefig(savefilename)



def create_plot(args):
    
    folder1_df, folder1_bf, folder2_df, folder2_bf = args.folders
    name1_ns, name1_s, name2_ns, name2_s = args.best_test_filenames

    filename1_df_ns =os.path.join(folder1_df,name1_ns)
    filename1_df_s =os.path.join(folder1_df,name1_s)
    filename1_bf_ns =os.path.join(folder1_bf,name1_ns)
    filename1_bf_s =os.path.join(folder1_bf,name1_s)

    filename2_df_ns =os.path.join(folder2_df,name2_ns)
    filename2_df_s =os.path.join(folder2_df,name2_s)
    filename2_bf_ns =os.path.join(folder2_bf,name2_ns)
    filename2_bf_s =os.path.join(folder2_bf,name2_s)

    data1_df_ns = read_csv(filename1_df_ns,matrix=True)
    data1_df_s = read_csv(filename1_df_s,matrix=True)
    data1_bf_ns = read_csv(filename1_bf_ns,matrix=True)
    data1_bf_s = read_csv(filename1_bf_s,matrix=True)

    data2_df_ns = read_csv(filename2_df_ns,matrix=True)
    data2_df_s = read_csv(filename2_df_s,matrix=True)
    data2_bf_ns = read_csv(filename2_bf_ns,matrix=True)
    data2_bf_s = read_csv(filename2_bf_s,matrix=True)

    data = data1_df_ns,data1_df_s,data1_bf_ns,data1_bf_s,data2_df_ns,data2_df_s,data2_bf_ns,data2_bf_s

    # combine_plot_line_with_error(data1, data2, args.datasets, args.folders, args.savefilename, args.experiment)
    combine_plot_line_with_error_same(data, args.datasets, args.folders, args.savefilename, args.experiment)
    

if __name__ == '__main__':

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', type=list_of_strings,
                        help="Folders housing the files we are needing.")
    parser.add_argument('--best_test_filenames', type=list_of_strings,
                        help="Name of files containing metrics for different simulations on\
                        the test partition of the datasets.")
    parser.add_argument('--datasets', type=list_of_strings, help="Name of datasets.")
    parser.add_argument('--exp', dest='experiment', default='eta',
                        help="Name of experiment (can be eta or walk_length).")
    parser.add_argument('--savefilename', default='zz_eta_vs_accuracy.png',
                        help="Filename of graph plotted.")
                        
    args = parser.parse_args()

    create_plot(args)
    print("Plot saved successfully!")

