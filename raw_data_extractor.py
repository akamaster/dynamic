import numpy as np
import pandas as pd

file = pd.read_csv('data/pa_initial.csv')

def string_processor(s):
    return ' '.join(s.split()).encode('ascii', errors='ignore').strip().decode('ascii')

non_normalized_initial_300 = [string_processor(line) for line in file['Variable']]
non_normalized_initial_300 = np.array(non_normalized_initial_300)

labels_initial_300 = np.array([x.strip() for x in file['context']])
labels_initial_300 = np.array(['fitness' if l in set(['fitness', 'active life', 'active life; fitness'])
        else 'environment' if l in set(['arrangement', 'barrier'])
                           else 'functional' for l in labels_initial_300])

print(non_normalized_initial_300.shape, labels_initial_300.shape)

def get_initial_data(file):
    non_normalized = []
    for line in file['Variable/Variable Question']:
        non_normalized.append(string_processor(line))

    return np.array(non_normalized)

def extract_dataset(file):
    non_normalized = get_initial_data(file)
    labels_full = np.array([' '*11]*len(non_normalized))
    labels = np.asarray(file['Activity'].values, dtype ='U')
    labels_full[labels!='nan'] = 'fitness'
    labels = np.asarray(file['Environment'].values, dtype ='U')
    labels_full[labels!='nan'] = 'environment'
    labels = np.asarray(file['Physical Function'].values, dtype ='U')
    labels_full[labels!='nan'] = 'functional'
    labels_idx = labels_full != ' '*11
    labels_full = labels_full[labels_idx]
    non_normalized = non_normalized[labels_idx]
    return non_normalized, labels_full

file = pd.read_csv('data/pa_annotations_HK.csv')
non_normalized_HK, labels_HK = extract_dataset(file)
print(labels_HK)
file = pd.read_csv('data/pa_annotations_JQ.csv')
non_normalized_JQ, labels_JQ = extract_dataset(file)

file = pd.read_csv('data/pa_annotations_JO.csv')
non_normalized_JO, labels_JO = extract_dataset(file)

file = pd.read_csv('data/nonactivity.csv')
non_normalized_nonactivity = []
for line in file['variable descriptions']:
    non_normalized_nonactivity.append(string_processor(line))

non_normalized_nonactivity_labels = np.array(['neither']*len(non_normalized_nonactivity))

text = np.hstack((non_normalized_initial_300,
                  non_normalized_HK,
                  non_normalized_JQ,
                  non_normalized_JO,
                  non_normalized_nonactivity))
labels = np.hstack((labels_initial_300,
                    labels_HK,
                    labels_JQ,
                    labels_JO,
                    non_normalized_nonactivity_labels))

data = np.vstack((text, labels))
print(data[1,:])
np.savetxt('data.csv', data.T, delimiter='|', newline='\n', fmt="%s", comments=None)
np.savez('data.npz', data.T)
print(data.shape)
