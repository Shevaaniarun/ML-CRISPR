import numpy as np

def gc_content(seq):
    """Compute GC percentage"""
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)

def nucleotide_counts(seq):
    """Count A,C,G,T"""
    return [
        seq.count("A"),
        seq.count("C"),
        seq.count("G"),
        seq.count("T")
    ]

def kmer_counts(seq, k=2):
    """Count all k-mer frequencies"""
    kmers = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmers[kmer] = kmers.get(kmer, 0) + 1
    return kmers

def extract_features(sequence_list):
    """
    Convert sgRNA sequences into numerical feature vectors.
    """

    feature_matrix = []
    all_kmers = set()

    # First pass: collect kmers
    for seq in sequence_list:
        all_kmers.update(kmer_counts(seq, k=2).keys())

    all_kmers = sorted(list(all_kmers))

    # Second pass: build feature vectors
    for seq in sequence_list:
        features = []

        # GC content
        features.append(gc_content(seq))

        # Nucleotide composition
        features.extend(nucleotide_counts(seq))

        # 2-mer frequencies
        kmer_freqs = kmer_counts(seq, k=2)
        for kmer in all_kmers:
            features.append(kmer_freqs.get(kmer, 0))

        feature_matrix.append(features)

    return np.array(feature_matrix)
