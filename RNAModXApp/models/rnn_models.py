from Bio import SeqIO
#
# from Bio import Entrez, SeqIO
#
# # Replace with your own email address
# Entrez.email = "your_email@example.com"
#
# # Replace "WASH7P" with the gene name you're interested in
# gene_name = "WASH7P"
#
# # Search for the gene in the Nucleotide database
# search_handle = Entrez.esearch(db="nucleotide", term=f"{gene_name} [GENE] AND Homo sapiens [ORGN]")
# search_record = Entrez.read(search_handle)
# search_handle.close()
#
# # Get the GenBank accession number of the first search result
# accession = search_record["IdList"][0]
#
# # Fetch the corresponding sequence
# fetch_handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
# seq_record = SeqIO.read(fetch_handle, "genbank")
# fetch_handle.close()
#
# # Print the RNA sequence
# for s in seq_record.seq:
#     print(s)
#     break
from Bio import SeqIO

length_limit = 100001
counter = 0
sequence = ""

for seq_record in SeqIO.parse("C://Users//shashi.vish//Downloads//GRCh38_latest_genomic.fna", "fasta"):
    data  = str(seq_record.seq)
    name = seq_record.name
    print(name , len(data))
    counter +=1
    print("Counter ---------------->" , counter)



print("----------------------")
print(sequence)
