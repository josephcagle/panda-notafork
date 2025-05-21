
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from scipy.integrate import trapezoid
import os


with open('panda3dcode/train_w_go_greater_50.pkl','rb') as f:
    go_occurrences = pickle.load(f)

panda_terms = sorted(set(go_occurrences.keys()))
print(f'Number of GO terms used by PANDA-3D (filter:50): {len(panda_terms)}')
print()

import pandas as pd
cafa_train_terms = pd.read_csv('../APF/Train/train_terms.tsv', sep='\t')
# print(f'Number of GO terms in CAFA 5 train dataset (filter:50): {len(set(cafa_train_terms["term"]))}')
cafa_train_terms_set_dict = cafa_train_terms.groupby("aspect")["term"].apply(set).to_dict()
# # print(cafa_train_terms_set_dict)
# everything = cafa_train_terms_set_dict['MFO'] | cafa_train_terms_set_dict['BPO'] | cafa_train_terms_set_dict['CCO']
# print(f"Number of GO terms in CAFA 5 dataset (filter:0): {len(everything)}")
# print(f"Number of MF terms in CAFA 5 dataset (filter:0): {len(cafa_train_terms_set_dict['MFO'])}")
# print()

# print(f"Number of terms in both PANDA-3D (filter:50) and CAFA 5 dataset (filter:0): {len(set(panda_terms) & set(everything))}")
# print(f"Number of MF terms in both PANDA-3D (filter:50) and CAFA 5 dataset (filter:0) (target set): {len(set(panda_terms) & set(cafa_train_terms_set_dict['MFO']))}")
# print()

target_terms = sorted(set(panda_terms) & set(cafa_train_terms_set_dict['MFO']))

# parse prediction.txt

def parse_prediction_file(prediction_file):
    with open(prediction_file, 'r') as f:
        lines = f.readlines()
    
    predictions = defaultdict(list)
    for line in lines[3:]:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        protein_id = parts[0].split('-')[1]
        go_term = parts[1]
        confidence = float(parts[2])
        predictions[protein_id].append((go_term, confidence))
    
    return predictions

predictions = parse_prediction_file('prediction.txt')

# print(f'Number of predictions for each protein: {({protein_id: len(predictions[protein_id]) for protein_id in predictions})}')
print(f'Average number of predictions per protein: {np.mean([len(preds) for preds in predictions.values()]):.2f}')
print()

avg_confidence = np.mean([confidence for preds in predictions.values() for _, confidence in preds])
print(f'Average confidence of predictions: {avg_confidence:.2f}')
print()

filtered_predictions = {
    protein_id: [(go_term, confidence) for go_term, confidence in preds if go_term in target_terms]
    for protein_id, preds in predictions.items()
}
# print(f'Number of filtered predictions for each protein: {({protein_id: len(filtered_predictions[protein_id]) for protein_id in filtered_predictions})}')
print(f'Average number of filtered predictions per protein: {np.mean([len(preds) for preds in filtered_predictions.values()]):.2f}')
print()


# filtered_proportions = [len(filtered_preds) / len(predictions[protein_id]) for protein_id, filtered_preds in filtered_predictions.items()]
# q1 = np.percentile(filtered_proportions, 25)
# print(f'Q1: {q1:.2f}')
# avg_filtered_proportion = np.mean(filtered_proportions)
# print(f'Average proportion of filtered predictions: {avg_filtered_proportion:.2f}')
# median_filtered_proportion = np.median(filtered_proportions)
# print(f'Median proportion of filtered predictions: {median_filtered_proportion:.2f}')
# q3 = np.percentile(filtered_proportions, 75)
# print(f'Q3: {q3:.2f}')


def get_filtered_preds_at_thresh(threshold):
    print(f"Getting filtered predictions at confidence threshold: {threshold}")
    thresh_filtered_predictions = {
        protein_id: [go_term for go_term, confidence in preds if confidence >= threshold]
        for protein_id, preds in filtered_predictions.items()
    }
    return thresh_filtered_predictions


print("Building CAFA MF training terms dictionary")
cafa_train_terms_mf_dict = defaultdict(list)
for _, row in tqdm(cafa_train_terms.iterrows(), total=cafa_train_terms.shape[0], desc="Processing CAFA training terms", smoothing=0):
    if row['aspect'] == 'MFO':
        cafa_train_terms_mf_dict[row['EntryID']].append(row['term'])

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

def compute_metrics(predictions):
    print(f"Computing metrics for predictions")
    if not predictions or not cafa_train_terms_mf_dict or not target_terms:
        raise ValueError("Empty input detected")

    protein_ids = list(predictions.keys())
    n_proteins = len(protein_ids)
    n_terms = len(target_terms)
    
    # Initialize NumPy arrays for true and predicted labels
    true_labels_array = np.zeros((n_proteins, n_terms), dtype=bool)
    pred_labels_array = np.zeros((n_proteins, n_terms), dtype=bool)
    
    # Vectorized label assignment, only considering target_terms
    for i, protein_id in tqdm(enumerate(protein_ids), total=n_proteins, desc="Processing proteins", smoothing=0):
        if protein_id not in cafa_train_terms_mf_dict:
            raise ValueError(f'Protein ID {protein_id} not found in CAFA training terms')
        
        preds = predictions[protein_id]
        true_terms = cafa_train_terms_mf_dict[protein_id]
        
        # Only target_terms are evaluated, ignoring other terms in predictions or cafa_train_terms_mf_dict
        true_labels_array[i] = np.array([term in true_terms for term in target_terms])
        pred_labels_array[i] = np.array([term in preds for term in target_terms])
    
    # Calculate F1 scores for each term
    term_f1_scores = {}
    for j, go_term in enumerate(target_terms):
        term_f1_scores[go_term] = f1_score(true_labels_array[:, j], pred_labels_array[:, j])
    
    # Calculate overall Fmax and AUPR
    precision, recall, _ = precision_recall_curve(true_labels_array.ravel(), pred_labels_array.ravel())
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    fmax = max(f1_scores) if f1_scores else 0
    aupr = trapezoid(recall, precision) if precision.size and recall.size else 0
    
    return {
        'term_f1_scores': term_f1_scores,
        'fmax': fmax,
        'aupr': aupr
    }

results = []
for threshold in [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    thresh_filtered_preds = get_filtered_preds_at_thresh(threshold)
    metrics = compute_metrics(thresh_filtered_preds)
    results.append({
        'threshold': threshold,
        **metrics
    })

# Print results
for result in results:
    print(f"Threshold: {result['threshold']}")
    print(f"Fmax: {result['fmax']:.4f}")
    print(f"AUPR: {result['aupr']:.4f}")
    # for term, f1 in result['term_f1_scores'].items():
    #     print(f"F1 score for {term}: {f1:.4f}")
    print()
