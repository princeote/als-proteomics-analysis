import pandas as pd

tsv_file = 'complement_genes.tsv'
csv_file = 'complement_genes.csv'

# 1. Read the TSV file, explicitly defining the separator as a tab ('\t')
df = pd.read_csv(tsv_file, sep='\t')

# 2. Write the data to a new CSV file (default separator is comma ',')
#    index=False prevents writing the DataFrame index as a column in the CSV
df.to_csv(csv_file, index=False)

print(f"Successfully converted '{tsv_file}' to '{csv_file}'")