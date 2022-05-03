from biotransformers import BioTransformers


bio_trans = BioTransformers(backend="protbert")

sequences = [
    "MKTVRQERLKSIVRILERSKEAVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]

loglikelihood = bio_trans.compute_loglikelihood(sequences, batch_size=2)
print(loglikelihood)


UNNATURAL = list("ACDEFGHIKLMNPQRSTVWY") + ["-"]
loglikelihood = bio_trans.compute_loglikelihood(sequences, tokens_list=UNNATURAL)
print(loglikelihood)


from biotransformers import BioTransformers

bio_trans = BioTransformers(backend="protbert")

sequence = ["MKT"]
probabilities = bio_trans.compute_probabilities(sequence)

print(probabilities)
