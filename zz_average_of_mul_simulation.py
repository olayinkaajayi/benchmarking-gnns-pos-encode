import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os

def save_float_to_csv(value, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if isinstance(value, list):
            writer.writerow(value)
        else:
            writer.writerow([value])

    csv_name = filename.split('/')[-1]
    print(f"{csv_name} file saved successfully!")


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


def plot_line_with_error(mean, std_dev, dataset, folder, exp='eta', train_data=True):
    if dataset == "WikiCS":
        # x = [1e4, 5e4, 8e4, 10e4, 11e4, 12e4, 5e5] if exp == 'eta' else [50, 80, 100, 120]
        x = [1e3, 3e3, 5e3, 7e3, 9e3, 1e4] if exp == 'eta' else [50, 80, 100, 120]
    else:
        x = [1e4, 5e4, 7e4, 1e5, 5e5, 7e5] if exp == 'eta' else [50, 80, 100, 120]

    plt.figure(int(train_data)+1)
    plt.plot(x, mean, marker='o', color='k')
    plt.fill_between(x, mean - std_dev, mean + std_dev, alpha=0.2)

    # Rescale x-axis
    ax = plt.gca()
    ax.set_xscale('log')

    plt.title('Ablation {}'.format('Training set' if train_data else 'Test data'))
    plt.xlabel(f'$\eta$' if exp=='eta' else 'Random walk length')
    plt.ylabel('@50 Hits (%)' if 'OGB' in dataset else 'Accuracy (%)')

    train_test = 'train' if train_data else 'test'
    filename = f'Ablation_{train_test}_{dataset}_{exp}.png'
    plt.savefig(os.path.join(folder,filename))


def combine_plot_line_with_error(data, dataset, folder, exp='eta'):
    if dataset == "WikiCS":
        x = [1e3, 3e3, 5e3, 7e3, 9e3, 1e4, 5e4, 8e4, 10e4, 11e4, 12e4, 5e5]
    elif dataset == "Pubmed":
        x = [1e3, 5e3, 1e4, 2e4, 5e4, 10e4, 15e4, 5e5]
    else:
        x = [1e4, 5e4, 7e4, 1e5, 5e5, 7e5] if exp == 'eta' else [50, 80, 100, 120]
        # No sign flip: probe the range (1e6, 5e6)
        # etas=(2e6 2.5e6 3e6 3.5e6 4e6 4.5e6)

    mean_train = data[:,0]
    std_train = data[:,1]
    mean_test = data[:,2]
    std_test = data[:,3]

    plt.figure()
    # Train data
    plt.plot(x, mean_train, marker='o', color='g', label="Train")
    plt.errorbar(x, mean_train, yerr = std_train, fmt ='o', color='g', capsize=3)

    # Test data
    plt.plot(x, mean_test, marker='o', color='brown', label="Test")
    plt.errorbar(x, mean_test, yerr = std_test, fmt ='o', color='brown', capsize=3)

    # Rescale x-axis
    ax = plt.gca()
    ax.set_xscale('log')
    # plt.ylim(82, 96)
    plt.ylim(np.min(mean_test)-np.max(std_test)-1, np.max(mean_train)+np.max(std_train)+1)
    plt.title(f'Effect of $\eta$ on Accuracy for {dataset}')
    plt.xlabel(f'$\eta$')
    plt.ylabel('@50 Hits (%)' if 'OGB' in dataset else 'Accuracy (%)')
    plt.legend()

    filename = f'Ablation_{dataset}_{exp}_experiment.png'
    plt.savefig(os.path.join(folder,filename))


def create_plot(args):
    # if args.dataset == "WikiCS":
    #     name = 'combine_average_metric_eta_wikics.csv'
    # else:
    name = f'Avg_metric_for_{args.experiment}_{args.dataset}.csv'
    filename=os.path.join(args.folder,name)
    data = read_csv(filename,matrix=True)

    if args.combine:
        combine_plot_line_with_error(data, args.dataset, args.folder, args.experiment)
    else:
        mean_train = data[:,0]
        std_train = data[:,1]
        plot_line_with_error(mean_train, std_train, args.dataset, args.folder, args.experiment)

        mean_test = data[:,2]
        std_test = data[:,3]
        plot_line_with_error(mean_test, std_test, args.dataset, args.folder, args.experiment, train_data=False)



def main(args):
    """
        We gather the metric (accuracy or @50Hits) for the different seeds of the
        dataset we work with. And we compute the average and S.D.

        args.best_acc_filename: should be a csv file. It is a file holding the best
                                metrics for many simulations on our dataset.
    """
    print("\n\n")
    filename_train = os.path.join(args.folder,args.best_train_filename)
    filename_test = os.path.join(args.folder,args.best_test_filename)

    scores_train = read_csv(filename=filename_train, matrix=args.dataset=="WikiCS")
    if isinstance(scores_train, np.ndarray):
        avg_train = np.mean(scores_train[:,0])
        # Note that for the WikiCS dataset, the algorithm is ran for 20 train-test split (cross-validation).
        # So we have the mean and std of the 20 splits. So we use this "suggested" formula to compute
        # the standard deviation of N simulations (different seeds) of the GNN algorithm on the dataset.
        std_train = np.sqrt(np.mean(scores_train[:,1]**2)  + np.var(scores_train[:,0]))
    else:
        avg_train = np.mean(scores_train)
        std_train = np.std(scores_train)

    print(f"Average (train) Best Score for {args.dataset} is {avg_train:.3f}+/-{std_train:.3f}")

    scores_test = read_csv(filename=filename_test, matrix=args.dataset=="WikiCS")
    if isinstance(scores_train, np.ndarray):
        avg_test = np.mean(scores_test[:,0])
        # Note that for the WikiCS dataset, the algorithm is ran for 20 train-test split (cross-validation).
        # So we have the mean and std of the 20 splits. So we use this "suggested" formula to compute
        # the standard deviation of N simulations (different seeds) of the GNN algorithm on the dataset.
        std_test = np.sqrt(np.mean(scores_test[:,1]**2)  + np.var(scores_test[:,0]))
    else:
        avg_test = np.mean(scores_test)
        std_test = np.std(scores_test)
    print(f"Average (test) Best Score for {args.dataset} is {avg_test:.3f}+/-{std_test:.3f}\n")

    if not args.simulation:
        with open(os.path.join(args.folder,f'Avg_metric_{args.dataset}.txt'),'w') as f:
            f.write(f"Statistics for Dataset: {args.dataset}\n\n")
            f.write(f"Average(train)={avg_train:.3f}\n")
            f.write(f"SD(train)={std_train:.3f}\n")

            f.write(f"Average(test)={avg_test:.3f}\n")
            f.write(f"SD(test)={std_test:.3f}\n")
            f.write("\n")
    else:
        save_float_to_csv([avg_train,std_train,avg_test,std_test],
                        filename=os.path.join(args.folder,f'Avg_metric_for_{args.experiment}_{args.dataset}.csv'))


    print("Saved average of metric...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='zz_ogb_collab_acc',
                        help="Folder housing the files we are needing.")
    parser.add_argument('--best_train_filename', default='train_@50Hits.csv',
                        help="Name of file containing metrics for different simulations on\
                        the train partition of the dataset.")
    parser.add_argument('--best_test_filename', default='test_@50Hits.csv',
                        help="Name of file containing metrics for different simulations on\
                        the test partition of the dataset.")
    parser.add_argument('--dataset', default='OGBL-COLLAB', help="Name of dataset.")
    parser.add_argument('--exp', dest='experiment', default='eta',
                        help="Name of experiment (can be eta or walk_length).")
    parser.add_argument('-s','--simulation', dest='simulation', action='store_true', default=False,
                        help='tells if we should save the mean & S.D as .csv file. (default: False)')
    parser.add_argument('-p','--plot', dest='get_plot', action='store_true', default=False,
                        help='tells if we should go ahead to create plot. (default: False)')
    parser.add_argument('-c','--combine', action='store_true', default=False,
                        help='tells if we should combine plots for both split of the dataset. (default: False)')

    args = parser.parse_args()

    if args.get_plot:
        create_plot(args)
        print("Plot saved successfully!")
    else:
        main(args)
