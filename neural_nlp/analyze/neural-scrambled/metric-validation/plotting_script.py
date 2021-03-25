import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from scipy import stats
from scipy.spatial import distance
import argparse

def get_normalized_dict(model_identifier, conditions, 
                        final_period, sentence_embedding,
                        norm_method, metric): #maybe pass activations dict as input so it doesn't have to recompute
    search_for = f'{sentence_embedding}_finalperiod={final_period}'
    activations_path = os.path.join(root_dir, 'model-activations', model_identifier)
    os.chdir(activations_path)

    for filename in os.listdir(activations_path):
        if search_for in filename and 'activations' in filename:
            with open(filename, 'rb') as f:
                layers_dic = pickle.load(f)
                break

    layer_cond_dict = {}
    for layer in list(layers_dic.keys()):
        layer_cond_dict[layer] = {}
    print(layer_cond_dict)
    
    mini = []
    maxi = []
    meani = []
    stdi = []
    layeri = []
    condi = []
    percenti = []
    modeli = []

    for cond in conditions:
    #print(cond)
        for filename in os.listdir(activations_path):
            if search_for in filename and 'activations' in filename:
                #print(filename)
                condition = filename.split("_")[2]
                #print(condition)
                if condition == cond:
                    print(condition)
                    print_cnt = 0
                    with open(filename, 'rb') as f:
                        data_dict = pickle.load(f)
                        #print(len(data_dict))
                        for layer in list(layer_cond_dict.keys()):
                            activations = list(data_dict[layer].values()) #list of lists, where each list is a sentence act.
                            
                            mini.append(np.min(activations))
                            maxi.append(np.max(activations))
                            meani.append(np.mean(activations))
                            stdi.append(np.std(activations))
                            layeri.append(layer)
                            condi.append(condition)
                            modeli.append(model_identifier)
                            mycount = ((np.array(activations) < -1) | (np.array(activations) > 1)).sum()
                            mypercent = float(np.round((mycount/np.array(activations).size)*100,2))
                            percenti.append(mypercent)
                        
                            if print_cnt == 0:
                                print(np.shape(activations))
                                print_cnt += 1
                            if norm_method == "clip":
                                #activations = np.random.rand(627,768)
                                activations = [np.clip(elm, -1,1) for elm in activations]
                            elif norm_method == "row_normalize":
                                activations = [stats.zscore(elm) for elm in activations]
                            elif norm_method == "no_norm":
                                activations = activations
                            else:
                                print("Normalization method not implemented! Using raw activations")
                            layer_cond_dict[layer][condition] = activations
    
    #############
    # save stats df
    #############
    df = pd.DataFrame(
    {'model': modeli,
     'layer': layeri,
     'condition': condi,
     'min': mini,
    'max': maxi,
     'mean': meani,
     'std': stdi,
    '(x<-1|x>1) in %': percenti})
    
    df_save_dir = os.path.join(root_dir, 'stats')
    df_savestr = os.path.join(df_save_dir,f"stats_{model_identifier}_{metric}_{sentence_embedding}_finalPeriod={final_period}_{norm_method}.csv")
    df.to_csv(df_savestr, index=False)

    return layer_cond_dict

def flatten_array(liste):
    liste_flatten = [item for sublist in liste for item in sublist]
    return liste_flatten


def get_correlations(layer_cond_dict, conditions, metric):
    plot_dict = {}
    for layer in tqdm(layer_cond_dict):
        correlations = []
        orig_column = flatten_array(layer_cond_dict[layer]['Original'])
        print(np.shape(orig_column))
            
        for cond in conditions:
            
            if metric == "spearman":
                corr = stats.spearmanr(orig_column, flatten_array(layer_cond_dict[layer][cond]))
                correlation = corr.correlation
            
            elif metric == "pearson":
                correlation = np.corrcoef(orig_column, flatten_array(layer_cond_dict[layer][cond]))[1,0]
            
            elif metric == "euclidean":
                # Euc
                all_euc = []
                nr_sentences = len(layer_cond_dict[layer]['Original'])
                for i in range(nr_sentences):
                    euc = distance.euclidean(layer_cond_dict[layer]['Original'][i], layer_cond_dict[layer][cond][i])
                    all_euc.append(euc)
                correlation = np.mean(all_euc)
            
            elif metric == "cosine":
                # Cos
                all_cos = []
                nr_sentences = len(layer_cond_dict[layer]['Original'])
                for i in range(nr_sentences):
                    if i%200 ==0:
                        print(i)
                    cos = distance.cosine(layer_cond_dict[layer]['Original'][i], layer_cond_dict[layer][cond][i])
                    all_cos.append(cos)
                correlation = np.mean(all_cos)
                
            correlations.append(correlation)
            
        plot_dict[layer] = correlations
        
    return plot_dict

def plot_correlations_lineplot(model_identifier, correlations_dict, conditions, metric,savestring):
    fig, ax = plt.subplots()
    line_colors = sns.color_palette("rocket") + sns.color_palette("GnBu_d") + [sns.color_palette("PRGn", 10)[2]] + [sns.color_palette("PuOr", 10)[0]]
    if model_identifier in ['xlnet-large-cased','bert-large-uncased-whole-word-masking']:
        line_colors = sns.color_palette("rocket") + sns.color_palette("GnBu_d") + sns.color_palette("PRGn", 10) + sns.color_palette("YlOrBr", 10)
    
    layers = list(correlations_dict.keys())
    
    counter = 0
    for key,value in correlations_dict.items():
        print(key, value)
        ax.plot(conditions,value, '-o',color=line_colors[counter])
        counter += 1

    #ax.set_title('Layer model activation correlation with model activations for unscrambled sentence across conditions: {}'.format(model_identifier))
    if not model_identifier in ['xlnet-large-cased', 'albert-xxlarge-v2', 'bert-large-uncased-whole-word-masking']:
        ax.legend(layers, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(layers, bbox_to_anchor=(1.05, 1), loc='upper left')
    if metric == "pearson":
        ax.yaxis.set_label_text('Pearson correlation')
        plt.ylim((0,1.05))
    elif metric == "spearman":
        ax.yaxis.set_label_text('Spearman correlation')
        plt.ylim((0,1.05))
    elif metric == "cosine":
        ax.yaxis.set_label_text('Cosine similarity')
        plt.ylim((-0.05,1))
    elif metric == "euclidean":
        ax.yaxis.set_label_text('L2 norm')
    plt.xticks(rotation= 90)
    plt.title(savestring.split("/")[-1].split(".")[0])
    
    plt.tight_layout()
    plt.savefig(savestring, dpi=240, bbox_inches='tight')

# final_period = "False"
# sentence_embedding = "last-tok"
# model_identifier = "gpt2"
# metric = "pearson"
# norm_method = "no_norm"
def plot_main(conditions, root_dir, model_identifier = "gpt2", final_period = "False", sentence_embedding = "last-tok", metric = "pearson",norm_method = "no_norm"):
    layer_cond_dict = get_normalized_dict(model_identifier, conditions, 
                        final_period, sentence_embedding,
                        norm_method, metric)
    correlations_dict = get_correlations(layer_cond_dict, conditions, metric)
    
    #############
    # save correlations dict
    #############
    corr_save_dir = os.path.join(root_dir, 'stats')
    corr_savestring = os.path.join(corr_save_dir,f"corrs_{model_identifier}_{metric}_{sentence_embedding}_finalPeriod={final_period}_{norm_method}.pkl")
    with open(corr_savestring, 'wb') as fout:
        pickle.dump(correlations_dict, fout)
    
    save_dir = os.path.join(root_dir, 'figs')
    savestring = os.path.join(save_dir,f"{model_identifier}_{metric}_{sentence_embedding}_finalPeriod={final_period}_{norm_method}.png")
    plot_correlations_lineplot(model_identifier, correlations_dict, conditions, metric,savestring)
    
    plot_correlations_lineplot(model_identifier, correlations_dict, conditions, metric,savestring)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description='plotting parser')
    parser.add_argument('--model_identifier', type=str) #default is 'avg-tok'
    parser.add_argument('--final_period', type=str) #True or False
    parser.add_argument('--sentence_embedding', type=str)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--norm_method', type=str)
    args = parser.parse_args()
    
    root_dir = '/om/user/ckauf/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/'
    conditions = ['Original', 'Scr1', 'Scr3', 'Scr5', 'Scr7', 'lowPMI', 'backward', 'random']
    
    plot_main(conditions=conditions, root_dir=root_dir,model_identifier=args.model_identifier, final_period=args.final_period,
              sentence_embedding=args.sentence_embedding, metric=args.metric, norm_method=args.norm_method)