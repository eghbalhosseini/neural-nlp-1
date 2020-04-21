import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime 
import seaborn
from stats import is_significant, _permutation_test
from scipy import stats


# Plot specifications
seaborn.set(context='talk')
seaborn.set_style("whitegrid", {'axes.grid': False})
plt.rc('axes', edgecolor='black')
plt.rc('axes', edgecolor='black')
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def getMeanLayerPref_perSubject(subjectID = '018',
                                score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl',
                                metric = 'mean',
                                categorize = False, # Split into early, mid, late
                                plotROIs = True,
                                plotNetworks = True):

    # Define variables and file names
    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
    fname = score_name.split('=')
    pngfile_name = subjectID + '_ROI_' + metric + '_LayerPref_' + timestamp + fname[1][:-5] + fname[2][:-10] + '.png'
    pngfile_name = pngfile_name.replace(',', '_')
    pngfile_name = pngfile_name.replace('-', '_')
    
    statsfile_name = subjectID + '_stats_' + timestamp + '_' + fname[2][:-10] + '.csv'
    
    if plotNetworks:
        pngfile_name2 = subjectID + '_Network_' + metric + '_LayerPref_' + timestamp + '_' + fname[1][:-5] + fname[2][:-10] + '.png'
        pngfile_name2 = pngfile_name2.replace(',', '_')
        pngfile_name2 = pngfile_name2.replace('-', '_')
    
    # Load score objects
    with open(score_name, 'rb') as f:  
        result = pickle.load(f)
    
    score = result['data']

    roi_names_all = score.raw.roi.values
    roi_names = np.unique(roi_names_all)

    center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
    s = center_score.raw[{'neuroid': [subject == subjectID for subject in score.raw['subject'].values]}] 
    
    model2 = s.mean(['split', 'experiment']) # Not grouped by neuroid
    
#    _, index_model = np.unique(model2['neuroid_id'], return_index = True) # Omitted: get all neuroIDs
#    model3 = model2.isel(neuroid = np.sort(index_model))
    
    # Squeeze the layer dim
    model4 = model2.squeeze()
    
    # Sort by unique neuroid_id
    model5 = model4.sortby('neuroid_id')
    
    # Remove NaNs for subject 018
    if np.isnan(model5).any():
        nan_sum = np.isnan(model5).sum()
        print(nan_sum)
        
        model6 = model5.fillna(0)
    else:
        model6 = model5
    
    # Fetch the layer preference 
    model7 = model6.argmax('layer', skipna=True) # Argmax over layer dimension
    
    # Normalize scores or categorize
    if not categorize:
        model7_norm = model7/model7.max()
    if categorize:
        model7_norm = model7
        # Categorize into early (0-16), mid (17-32) and late (33-48) layers
        model7_norm[model7_norm <= 16] = 1
        model7_norm[(model7_norm >= 17) & (model7_norm <= 32)] = 2 
        model7_norm[model7_norm >= 33] = 3
 
    # Group by ROIs and average across ROIs 
    d = {}
    for roiID in roi_names:
        model7_roi = model7_norm[{'neuroid': [roi == roiID for roi in model7_norm['roi'].values]}]
        
        unique_atlas = np.unique(model7_roi.atlas.values)
        
        if len(unique_atlas) > 1:

            # Disentangle LH/RH AntTemp for lang and DMN
            model7_roi_lang = model7_roi[{'neuroid': [atlas == 'language' for atlas in model7_roi['atlas'].values]}]
            model7_roi_dmn = model7_roi[{'neuroid': [atlas == 'DMN' for atlas in model7_roi['atlas'].values]}]
            
            roiID_lang = roiID + '_lang'
            roiID_dmn = roiID + '_DMN'
            
            # Split into language voxels
            if metric == 'mean':
                roi_metric_lang = model7_roi_lang.mean().values
            if metric == 'median':
                roi_metric_lang = model7_roi_lang.median().values
            if metric == 'mode':
                roi_metric_lang = stats.mode(model7_roi_lang.values)
                roi_metric_lang = roi_metric_lang[0][0]
            roi_std_lang = model7_roi_lang.std().values
            d[roiID_lang + '_metric'] = [roi_metric_lang, roi_std_lang, len(model7_roi_lang.values), model7_roi_lang.values]

            # Split into DMN voxels
            if metric == 'mean':
                roi_metric_dmn = model7_roi_dmn.mean().values
            if metric == 'median':
                roi_metric_dmn = model7_roi_dmn.median().values
            if metric == 'mode':
                roi_metric_dmn = stats.mode(model7_roi_dmn.values)
                roi_metric_dmn = roi_metric_dmn[0][0]
            roi_std_dmn = model7_roi_dmn.std().values
            d[roiID_dmn + '_metric'] = [roi_metric_dmn, roi_std_dmn, len(model7_roi_dmn.values), model7_roi_dmn.values]
        
        if len(unique_atlas) == 1:
            if metric == 'mean':
                roi_metric = model7_roi.mean().values
            if metric == 'median':
                roi_metric = model7_roi.median().values
            if metric == 'mode':
                roi_metric = stats.mode(model7_roi.values)
                roi_metric = roi_metric[0][0]
            roi_std = model7_roi.std().values
            d[roiID + '_metric'] = [roi_metric, roi_std, len(model7_roi.values), model7_roi.values]

    ROInames = []
    ROIvalues = []
    for item in d.items():
        kval = item[0]
        sval = item[1]
        ROInames.append(kval)
        ROIvalues.append(sval)
    
    if plotROIs:
        names_ordered = ['LH_PostTemp', 'LH_AntTemp_lang', 'LH_IFG', 'LH_IFGorb', 'LH_MFG', \
                'LH_AngG', 'RH_PostTemp', 'RH_AntTemp_lang', 'RH_IFG', 'RH_IFGorb', \
                 'RH_MFG', 'RH_AngG', 'LH_postParietal', 'LH_midParietal', 'LH_antParietal', \
                 'LH_supFrontal', 'LH_Precentral_A_PrecG', 'LH_Precental_B_IFGop', 'LH_midFrontal', \
                 'LH_midFrontalOrb', 'LH_insula', 'LH_medialFrontal', 'RH_postParietal', \
                 'RH_midParietal', 'RH_antParietal', 'RH_supFrontal', 'RH_Precentral_A_PrecG', \
                 'RH_Precental_B_IFGop', 'RH_midFrontal', 'RH_midFrontalOrb', 'RH_insula', 'RH_medialFrontal', \
                 'LH_FrontalMed', 'LH_PostCing', 'LH_TPJ', 'LH_MidCing', 'LH_STGorInsula', \
                 'LH_AntTemp_DMN', 'RH_FrontalMed', 'RH_PostCing', 'RH_TPJ', 'RH_MidCing', 'RH_STGorInsula', 'RH_AntTemp_DMN', \
                 'LH_TE11', 'LH_TE12', 'RH_TE11', 'RH_TE12', 'LH_OccipInf', 'LH_OccipMid', 'LH_OccipSup', \
                 'RH_OccipInf', 'RH_OccipMid', 'RH_OccipSup'] 
        
        ROInames_plot = [name[ : -7] for name in ROInames]
        
        # Sort the ROI names to the list above (intended ROI order)
        name_score = list(zip(ROInames_plot, ROIvalues))
        name_score_sorted = name_score.copy()
        name_score_sorted.sort(key=lambda x: names_ordered.index(x[0]))
        
        names = [item[0] for item in name_score_sorted] 
        scores = [item[1][0] for item in name_score_sorted] 
        stds = [item[1][1] for item in name_score_sorted] 
        brain_scores = [item[1][3] for item in name_score_sorted] # this is layer preference scores
        
        # Remove suffix from names for plotting
        names[1] = 'LH_AntTemp'
        names[7] = 'RH_AntTemp'
        names[37] = 'LH_AntTemp'
        names[43] = 'RH_AntTemp'
        
        names_new = []
        for name in names:
            name_new = name.replace('_', ' ')
            substr = name_new[2:4]
            substr_up = substr.upper()
            name_new_up = name_new.replace(substr, substr_up)
            names_new.append(name_new_up)

        color_lst = [(241/255, 150/255, 135/255)]*12 + [(240/255, 211/255, 128/255)]*20 + \
        [(136/255, 192/255, 133/255)]*12 + ['lightsteelblue']*4 + ['thistle']*6
        
        xticks = list(range(len(names)))
        xtick_lst = [x + .33 for x in xticks] 
        
        plt.figure(figsize=(16, 8))
        plt.bar(range(len(names)), scores, yerr = stds, error_kw = dict(lw=2, capsize=0, capthick=1), \
                ecolor='black', align='edge', width=0.7, color = color_lst)
        plt.xticks(xtick_lst, names_new, size='small', rotation=90)
        plt.title('GPT2-xl - Normalized Layer Preference Across ROIs (' + metric + '), Subject ' + subjectID)
        plt.ylabel('Layer Number', size='small') 
        plt.ylim(0,1.2)
        plt.tight_layout()
        plt.savefig(pngfile_name, dpi=240)

    # For plotting networks
    if plotNetworks:
        
        # Define all neuroIDs within each network
        lang_values = [i for subl in brain_scores[0:12] for i in subl]
        md_values = [i for subl in brain_scores[12:32] for i in subl]
        dmn_values = [i for subl in brain_scores[32:44] for i in subl]
        aud_values = [i for subl in brain_scores[44:48] for i in subl]
        vis_values = [i for subl in brain_scores[48:] for i in subl]
        
        lang_score = np.mean(lang_values)
        lang_std = np.std(lang_values)
        
        md_score = np.mean(md_values)
        md_std = np.std(md_values)
        
        dmn_score = np.mean(dmn_values)
        dmn_std = np.std(dmn_values)
        
        aud_score = np.mean(aud_values)
        aud_std = np.std(aud_values)
        
        vis_score = np.mean(vis_values)
        vis_std = np.std(vis_values)
        
        # Plot
        plt.figure(figsize=(7, 14)) 
        plt.bar([0, .2, .4, .6, .8], [lang_score, md_score, dmn_score, aud_score, vis_score], \
        yerr = [lang_std, md_std, dmn_std, aud_std, vis_std], \
        error_kw = dict(lw=2, capsize=0, capthick=1), ecolor='black', align='edge', \
        width = .1, color = [(241/255, 150/255, 135/255), (240/255, 211/255, 128/255), \
                             (136/255, 192/255, 133/255), 'lightsteelblue', 'thistle'])
        plt.xticks([0.05, .25, .45, .65, .85], \
           ['Language', 'Multiple Demand', 'Default Mode', 'Auditory', 'Visual'], \
           rotation=90, size='medium')
        plt.title('Normalized Layer Preference \n Across Networks (' + metric +'), Subject ' + subjectID)
        plt.ylabel('Layer Number', size='medium') # 'Layer Number'
        plt.ylim(0, 1.2)
        plt.tight_layout()
        plt.savefig(pngfile_name2, dpi=240)

