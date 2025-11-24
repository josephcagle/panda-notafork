
from Bio import SeqIO

# Read list of terms that have been filtered by PSI-BLAST similarity
filtered_fasta = '/home/joseph/code/APF/ALPHAFOLDpipeline/psiblastfiltering/test_filtered_60.fasta'
filtered_ids = set()
for i, record in enumerate(SeqIO.parse(filtered_fasta, 'fasta')):
    filtered_ids.add(record.id)
print(f"Read {len(filtered_ids)} filtered uniprot ids from {filtered_fasta}")

predictions_tsv = "prediction.txt"
outfile = "prediction_filtered_0.6.txt"

with open(predictions_tsv, 'r') as f, open(outfile, 'w') as out_f:
    # copy first 3 lines
    for _ in range(3):
        out_f.write(next(f))

    num_in_filtered = 0
    num_total = 0
    for line in f:
        num_total += 1
        uniprot_id = line.split('\t')[0].removeprefix("AF-").removesuffix("-F1-model_v4")
        if uniprot_id in filtered_ids:
            out_f.write(line)
            num_in_filtered += 1

    print(f"Read {len(filtered_ids)} filtered uniprot ids from {filtered_fasta}")
    print(f"Processed {num_total} rows, found {num_in_filtered} predictions in filtered set ({num_in_filtered / num_total * 100:.2f}%)")
    print(f"Filtered predictions saved to {outfile}")


