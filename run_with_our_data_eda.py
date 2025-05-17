
import pickle

with open('panda3dcode/train_w_go_greater_50.pkl','rb') as f:
    go_occurrences = pickle.load(f)

panda_terms = sorted(set(go_occurrences.keys()))
print(f'Number of GO terms used by PANDA-3D (filter:50): {len(panda_terms)}')
print()

# cafa_usable_go_terms = {'mf':[], 'bp':[], 'cc':[]}
# with open('../APF/ALPHAFOLDpipeline/usable_go_terms.txt','r') as f:
#     for line in f:
#         if line.endswith('MFO\n'):
#             cafa_usable_go_terms['mf'].append(line.split(' ')[0])
#         elif line.endswith('BPO\n'):
#             cafa_usable_go_terms['bp'].append(line.split(' ')[0])
#         elif line.endswith('CCO\n'):
#             cafa_usable_go_terms['cc'].append(line.split(' ')[0])
#         else:
#             raise ValueError(f'Unknown ontology: {line}')
# cafa_usable_go_terms_all = cafa_usable_go_terms['mf'] + cafa_usable_go_terms['bp'] + cafa_usable_go_terms['cc']
# print(f'Number of usable GO terms from CAFA 5 dataset (filter:25): {len(cafa_usable_go_terms_all)}')
# print(f'Number of MF usable GO terms from CAFA 5 dataset (filter:25): {len(cafa_usable_go_terms["mf"])}')
# print(f'Number of BP usable GO terms from CAFA 5 dataset (filter:25): {len(cafa_usable_go_terms["bp"])}')
# print(f'Number of CC usable GO terms from CAFA 5 dataset (filter:25): {len(cafa_usable_go_terms["cc"])}')

# with open('../APF/ALPHAFOLDpipeline/usable_mf_terms.txt','r') as f:
#     cafa5_mf_terms = f.read().splitlines()
# print(f'Number of usable MF GO terms from CAFA 5 dataset (which we have used - filter:50): {len(cafa5_mf_terms)}')
# print()

# print(f"Number of terms in both PANDA-3D (filter:50) and usable_go_terms (filter:25): {len(set(panda_terms) & set(cafa_usable_go_terms_all))}")
# print(f"Number of MF terms in both PANDA-3D (filter:50) and usable_go_terms (filter:25): {len(set(panda_terms) & set(cafa5_mf_terms))}")
# print(f"Number of BP terms in both PANDA-3D (filter:50) and usable_go_terms (filter:25): {len(set(panda_terms) & set(cafa_usable_go_terms['bp']))}")
# print(f"Number of CC terms in both PANDA-3D (filter:50) and usable_go_terms (filter:25): {len(set(panda_terms) & set(cafa_usable_go_terms['cc']))}")
# print(f"Number of terms in both PANDA-3D (filter:50) and usable_mf_terms (filter:50): {len(set(panda_terms) & set(cafa5_mf_terms))}")

import pandas as pd
cafa_train_terms = pd.read_csv('../APF/Train/train_terms.tsv', sep='\t')
# print(f'Number of GO terms in CAFA 5 train dataset (filter:50): {len(set(cafa_train_terms["term"]))}')
cafa_train_terms_dict = cafa_train_terms.groupby("aspect")["term"].apply(set).to_dict()
# print(cafa_train_terms_dict)
everything = cafa_train_terms_dict['MFO'] | cafa_train_terms_dict['BPO'] | cafa_train_terms_dict['CCO']
print(f"Number of GO terms in CAFA 5 dataset (filter:0): {len(everything)}")
print(f"Number of MF terms in CAFA 5 dataset (filter:0): {len(cafa_train_terms_dict['MFO'])}")
print()

print(f"Number of terms in both PANDA-3D (filter:50) and CAFA 5 dataset (filter:0): {len(set(panda_terms) & set(everything))}")
print(f"Number of MF terms in both PANDA-3D (filter:50) and CAFA 5 dataset (filter:0): {len(set(panda_terms) & set(cafa_train_terms_dict['MFO']))}")








# import run_PANDA3D
# run_PANDA3D.main()


