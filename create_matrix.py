import time
import numpy as np
import pandas as pd

path = 'data/cut_kmer/'
kmers = open("columns.txt").read().split()
files = open("rows.txt").read().split()

kmers_dictionary = {kmer: i for i, kmer in enumerate(kmers)}
matrix = np.zeros((len(files), len(kmers)))

for i, file in enumerate(files):
    start = time.time()
    data = open(path + file, encoding="utf8").read().split("\n")
    values = data[::2]
    current_values = values[:-1]
    current_kmers = data[1::2]
    assert(len(current_values) == len(current_kmers))
    current_values = list(map(lambda x: int(x.replace(">","")), current_values))
    for k, kmer in enumerate(current_kmers):
        j = kmers_dictionary[kmer]
        matrix[i][j] = current_values[k]
    print(i, file)
    print(time.time() - start)
    
    
klebseilla_gentamicin = "data/Klebseilla_gentamicin.csv"
data = pd.read_csv(klebseilla_gentamicin)
short_data = data[["Antibiotic", "Measurement.Sign", "Measurement.Value", "SRA.Accession"]]

srr_names = []
for file in filenames:
    srr_names += re.findall("SRR\d+", file)
srr_names = list(set(srr_names))

short_data = short_data.loc[short_data['SRA.Accession'].isin(srr_names)]
srr2mic = {srr : mic for srr, mic in zip(list(short_data['SRA.Accession']), list(short_data['Measurement.Value']))}

identificators = [file[:10] for i, file in enumerate(files)]
mic = np.array([srr2mic[ind] for ind in identificators])
mic = mic.reshape((-1, 1))


# temp = np.copy(matrix[:, :-1])
# temp[temp > 0] = 1
# binary_matrix = np.hstack((temp, mic))

count_matrix = np.hstack((matrix, mic))

np.savetxt("matrix_.csv", count_matrix, delimiter=",")
# np.savetxt("binary_matrix_.csv", binary_matrix, delimiter=",")