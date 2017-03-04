import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main():
    #ndcg_path = 'D:\cours\MA1\Semester Project\datasets\holidays\outputs\\ndcg\\'
    ndcg_path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\outputs\\ndcg\\'
    ks = [10, 15, 20,25, 30, 40, 50]
    for k in ks:
        print '##########################################'
        print '################# K='+str(k)+' ######################'
        print '##########################################'

        csv_path = ndcg_path+str(k)+'k_ndcg_metrics.csv'

        bar_plot_for_k(csv_path)

def bar_plot_for_k(filepath):
    sns.set_style("whitegrid")
    dmetrics = pd.read_csv(filepath)
    dmetrics.columns = ['img_name', 'feature', 'algorithm', 'ndcg']
    data = sns.load_dataset("titanic")
    sns.barplot(x="feature", y="ndcg", hue="algorithm", data=dmetrics)
    plt.show()

main()
