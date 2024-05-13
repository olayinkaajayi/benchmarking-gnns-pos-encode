import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os



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
    plt.plot(x1, mean_test1, marker='o', color='g', label="WikiCS")
    plt.errorbar(x1, mean_test1, yerr = std_test1, fmt ='o', color='g', capsize=3)

    # Pubmed
    plt.plot(x2, mean_test2, marker='o', color='brown', label="PubMed")
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

    data1,data2,data3,data4 = data
    
    # WikiCS: no-sign
    mean_test1 = data1[:,2]
    std_test1 = data1[:,3]
    # WikiCS: sign
    mean_test2 = data2[:,2]
    std_test2 = data2[:,3]

    #Pubmed: no-sign
    mean_test3 = data3[:,2]
    std_test3 = data3[:,3]
    #Pubmed: sign
    mean_test4 = data4[:,2]
    std_test4 = data4[:,3]

    # Create a figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2 , figsize=(10, 10))

    # Plot 1 and 2
    # WikiCS: no-sign
    ax1.plot(wiki, mean_test1, marker='o', color='g', label="WikiCS NO sign flip")
    ax1.errorbar(wiki, mean_test1, yerr = std_test1, fmt ='o', color='g', capsize=3)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel(f'$\eta$')
    # WikiCS: no-sign
    ax2.plot(wiki, mean_test2, marker='o', color='r', label="WikiCS sign flip")
    ax2.errorbar(wiki, mean_test2, yerr = std_test2, fmt ='o', color='r', capsize=3)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlabel(f'$\eta$')
    

    # Plot 3 and 4
    # Pubmed: no-sign
    ax3.plot(pub, mean_test3, marker='o', color='b', label="PubMed NO sign flip")
    ax3.errorbar(pub, mean_test3, yerr = std_test3, fmt ='o', color='b', capsize=3)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_xlabel(f'$\eta$')
    # Pubmed: no-sign
    ax4.plot(pub, mean_test4, marker='o', color='k', label="PubMed sign flip")
    ax4.errorbar(pub, mean_test4, yerr = std_test4, fmt ='o', color='k', capsize=3)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xlabel(f'$\eta$')

    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    wiki_m = np.concatenate([mean_test1,mean_test2])
    wiki_s = np.concatenate([std_test1,std_test2])

    pub_m = np.concatenate([mean_test3,mean_test4])
    pub_s = np.concatenate([std_test3,std_test4])

    ax1.set_ylim(np.min(wiki_m)-np.max(wiki_s)-1, np.max(wiki_m)+np.max(wiki_s)+1)
    ax2.set_ylim(np.min(wiki_m)-np.max(wiki_s)-1, np.max(wiki_m)+np.max(wiki_s)+1)
    ax3.set_ylim(np.min(pub_m)-np.max(pub_s)-1, np.max(pub_m)+np.max(pub_s)+1)
    ax4.set_ylim(np.min(pub_m)-np.max(pub_s)-1, np.max(pub_m)+np.max(pub_s)+1)

    # ax1.set_ylim(np.min(mean_test1)-np.max(std_test1)-1, np.max(mean_test1)+np.max(std_test1)+1)
    # ax2.set_ylim(np.min(mean_test2)-np.max(std_test2)-1, np.max(mean_test2)+np.max(std_test2)+1)
    # ax3.set_ylim(np.min(mean_test3)-np.max(std_test3)-1, np.max(mean_test3)+np.max(std_test3)+1)
    # ax4.set_ylim(np.min(mean_test4)-np.max(std_test4)-1, np.max(mean_test4)+np.max(std_test4)+1)

    fig.suptitle(f'Effect of $\eta$ on Accuracy for {datasets[0]} and {datasets[1]} datasets')

    # Adjust layout
    plt.tight_layout()

    # save the final image
    plt.savefig(savefilename)



def create_plot(args):
    
    folder1, folder2 = args.folders
    name1, name2, name3, name4 = args.best_test_filenames
    filename1 =os.path.join(folder1,name1)
    filename2 =os.path.join(folder1,name2)
    filename3 =os.path.join(folder2,name3)
    filename4 =os.path.join(folder2,name4)
    data1 = read_csv(filename1,matrix=True)
    data2 = read_csv(filename2,matrix=True)
    data3 = read_csv(filename3,matrix=True)
    data4 = read_csv(filename4,matrix=True)

    data = data1,data2,data3,data4

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

