from Bio import SeqIO
import sys

sys.stdout = open("test.csv", "w")
for re in SeqIO.parse("test.fasta", "fasta"):
    print(">{}\t{}".format(str(re.description), re.seq))


sys.stdout.close()
