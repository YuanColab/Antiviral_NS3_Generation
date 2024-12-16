from modlamp.analysis import GlobalAnalysis
from modlamp.core import count_aas
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from modlamp.sequences import Random, Helices
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import numpy as np
import random


def _onehotencode(s, vocab=None):
    if vocab is None:
        vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', ' ']

    to_one_hot = np.eye(len(vocab), dtype=int)
    vocab_dict = {char: code for char, code in zip(vocab, to_one_hot)}

    result = [vocab_dict[l] for l in s]
    result = np.array(result)
    return np.reshape(result, (1, result.shape[0], result.shape[1])), vocab_dict, vocab

class SequenceHandler(object):
    def __init__(self, window=0, step=2, refs=True):
        self.sequences = None
        self.generated = None
        self.ran = None
        self.hel = None
        self.X = list()
        self.y = list()
        self.window = window
        self.step = step
        self.refs = refs
        _, self.to_one_hot, self.vocab = _onehotencode('A')

    def load_sequences(self, filename):
        with open(filename) as f:
            self.sequences = [s.strip() for s in f]
        self.sequences = random.sample(self.sequences, len(self.sequences))  # shuffle sequences randomly

    def pad_sequences(self, pad_char=' ', padlen=0):
        if pad_char not in self.vocab:
            self.vocab += [pad_char]

        if padlen:
            padded_seqs = [seq + pad_char * (self.step + self.window - len(seq)) if len(seq) < self.window else seq + pad_char * padlen for seq in self.sequences]
        else:
            length = max([len(seq) for seq in self.sequences])
            padded_seqs = ['B' + seq + pad_char * (length - len(seq)) for seq in self.sequences]

        self.sequences = padded_seqs

    def one_hot_encode(self, target='all'):
        if self.window == 0:
            self.X = np.array([[self.to_one_hot[char] for char in s[:-self.step]] for s in self.sequences])
            self.X = self.X.reshape(len(self.X), len(self.sequences[0]) - self.step, len(self.vocab))
            if target == 'all':
                self.y = np.array([[self.to_one_hot[char] for char in s[self.step:]] for s in self.sequences])
                self.y = self.y.reshape(len(self.y), len(self.sequences[0]) - self.step, len(self.vocab))
            elif target == 'one':
                self.y = [s[-self.step:] for s in self.sequences]

        else:
            self.X = np.array([[self.to_one_hot[char] for char in s[i: i + self.window]] for s in self.sequences for i in range(0, len(s) - self.window, self.step)])
            self.X = self.X.reshape(len(self.X), self.window, len(self.vocab))
            if target == 'all':
                self.y = np.array([[self.to_one_hot[char] for char in s[i + 1: i + self.window + 1]] for s in self.sequences for i in range(0, len(s) - self.window, self.step)])
                self.y = self.y.reshape(len(self.y), self.window, len(self.vocab))
            elif target == 'one':
                self.y = [s[-self.step:] for s in self.sequences]

        print("\nData shape:\nX: " + str(self.X.shape) + "\ny: " + str(np.array(self.y).shape))

    def analyze_training(self):
        d = GlobalDescriptor(self.sequences)
        d.length()
        print("\nLength distribution of pretraining data:\n")
        print("Number of sequences:    \t%i" % len(self.sequences))
        print("Mean sequence length:   \t%.1f ± %.1f" % (np.mean(d.descriptor), np.std(d.descriptor)))
        print("Median sequence length: \t%i" % np.median(d.descriptor))
        print("Minimal sequence length:\t%i" % np.min(d.descriptor))
        print("Maximal sequence length:\t%i" % np.max(d.descriptor))

    def analyze_generated(self, num, fname='analysis.txt', plot=False):
        with open(fname, 'w') as f:
            print("Analyzing...")
            f.write("ANALYSIS OF SAMPLED SEQUENCES\n==============================\n\n")
            f.write(
                "Number of duplicates in generated sequences: %i\n" % (len(self.generated) - len(set(self.generated))))
            count = len(set(self.generated) & set(self.sequences))  # get shared entries in both lists
            f.write("%.1f percent of generated sequences are present in the training data.\n" %
                    ((count / len(self.generated)) * 100))
            d = GlobalDescriptor(self.generated)
            len1 = len(d.sequences)
            d.filter_aa('B')
            len2 = len(d.sequences)
            d.length()
            f.write("\n\nLENGTH DISTRIBUTION OF GENERATED DATA:\n\n")
            f.write("Number of sequences too short:\t%i\n" % (num - len1))
            f.write("Number of invalid (with 'B'):\t%i\n" % (len1 - len2))
            f.write("Number of valid unique seqs:\t%i\n" % len2)
            f.write("Mean sequence length:     \t\t%.1f ± %.1f\n" % (np.mean(d.descriptor), np.std(d.descriptor)))
            f.write("Median sequence length:   \t\t%i\n" % np.median(d.descriptor))
            f.write("Minimal sequence length:  \t\t%i\n" % np.min(d.descriptor))
            f.write("Maximal sequence length:  \t\t%i\n" % np.max(d.descriptor))

            descriptor = 'pepcats'
            valid_sequences = [s for s in self.sequences if len(s) >= 4]
            seq_desc = PeptideDescriptor([s[1:].rstrip() for s in valid_sequences], descriptor)
            seq_desc.calculate_autocorr(4)
            gen_desc = PeptideDescriptor(d.sequences, descriptor)
            gen_desc.calculate_autocorr(4)

            # random comparison set
            self.ran = Random(len(self.generated), np.min(d.descriptor), np.max(d.descriptor))  # generate rand seqs
            probas = count_aas(''.join(seq_desc.sequences)).values()  # get the aa distribution of training seqs
            self.ran.generate_sequences(proba=probas)
            ran_desc = PeptideDescriptor(self.ran.sequences, descriptor)
            ran_desc.calculate_autocorr(4)

            # amphipathic helices comparison set
            self.hel = Helices(len(self.generated), np.min(d.descriptor), np.max(d.descriptor))
            self.hel.generate_sequences()
            hel_desc = PeptideDescriptor(self.hel.sequences, descriptor)
            hel_desc.calculate_autocorr(4)

            # distance calculation
            f.write("\n\nDISTANCE CALCULATION IN '%s' DESCRIPTOR SPACE\n\n" % descriptor.upper())
            desc_dist = distance.cdist(gen_desc.descriptor, seq_desc.descriptor, metric='euclidean')
            f.write("Average euclidean distance of sampled to training data:\t%.3f +/- %.3f\n" %
                    (np.mean(desc_dist), np.std(desc_dist)))
            ran_dist = distance.cdist(ran_desc.descriptor, seq_desc.descriptor, metric='euclidean')
            f.write("Average euclidean distance if randomly sampled seqs:\t%.3f +/- %.3f\n" %
                    (np.mean(ran_dist), np.std(ran_dist)))
            hel_dist = distance.cdist(hel_desc.descriptor, seq_desc.descriptor, metric='euclidean')
            f.write("Average euclidean distance if amphipathic helical seqs:\t%.3f +/- %.3f\n" %
                    (np.mean(hel_dist), np.std(hel_dist)))

            # more simple descriptors
            g_seq = GlobalDescriptor(seq_desc.sequences)
            g_gen = GlobalDescriptor(gen_desc.sequences)
            g_ran = GlobalDescriptor(ran_desc.sequences)
            g_hel = GlobalDescriptor(hel_desc.sequences)
            g_seq.calculate_all()
            g_gen.calculate_all()
            g_ran.calculate_all()
            g_hel.calculate_all()
            sclr = StandardScaler()
            sclr.fit(g_seq.descriptor)
            f.write("\n\nDISTANCE CALCULATION FOR SCALED GLOBAL DESCRIPTORS\n\n")
            desc_dist = distance.cdist(sclr.transform(g_gen.descriptor), sclr.transform(g_seq.descriptor),
                                       metric='euclidean')
            f.write("Average euclidean distance of sampled to training data:\t%.2f +/- %.2f\n" %
                    (np.mean(desc_dist), np.std(desc_dist)))
            ran_dist = distance.cdist(sclr.transform(g_ran.descriptor), sclr.transform(g_seq.descriptor),
                                      metric='euclidean')
            f.write("Average euclidean distance if randomly sampled seqs:\t%.2f +/- %.2f\n" %
                    (np.mean(ran_dist), np.std(ran_dist)))
            hel_dist = distance.cdist(sclr.transform(g_hel.descriptor), sclr.transform(g_seq.descriptor),
                                      metric='euclidean')
            f.write("Average euclidean distance if amphipathic helical seqs:\t%.2f +/- %.2f\n" %
                    (np.mean(hel_dist), np.std(hel_dist)))

            # hydrophobic moments
            uh_seq = PeptideDescriptor(seq_desc.sequences, 'eisenberg')
            uh_seq.calculate_moment()
            uh_gen = PeptideDescriptor(gen_desc.sequences, 'eisenberg')
            uh_gen.calculate_moment()
            uh_ran = PeptideDescriptor(ran_desc.sequences, 'eisenberg')
            uh_ran.calculate_moment()
            uh_hel = PeptideDescriptor(hel_desc.sequences, 'eisenberg')
            uh_hel.calculate_moment()
            f.write("\n\nHYDROPHOBIC MOMENTS\n\n")
            f.write("Hydrophobic moment of training seqs:\t%.3f +/- %.3f\n" %
                    (np.mean(uh_seq.descriptor), np.std(uh_seq.descriptor)))
            f.write("Hydrophobic moment of sampled seqs:\t\t%.3f +/- %.3f\n" %
                    (np.mean(uh_gen.descriptor), np.std(uh_gen.descriptor)))
            f.write("Hydrophobic moment of random seqs:\t\t%.3f +/- %.3f\n" %
                    (np.mean(uh_ran.descriptor), np.std(uh_ran.descriptor)))
            f.write("Hydrophobic moment of amphipathic seqs:\t%.3f +/- %.3f\n" %
                    (np.mean(uh_hel.descriptor), np.std(uh_hel.descriptor)))

        if plot:
            if self.refs:
                a = GlobalAnalysis([uh_seq.sequences, uh_gen.sequences, uh_hel.sequences, uh_ran.sequences],
                                   ['training', 'sampled', 'hel', 'ran'])
            else:
                a = GlobalAnalysis([uh_seq.sequences, uh_gen.sequences], ['training', 'sampled'])
            a.plot_summary(filename=fname[:-4] + '.png')

    def save_generated(self, logdir, filename):
        with open(filename, 'w') as f:
            for s in self.generated:
                f.write(s + '\n')

        self.ran.save_fasta(logdir + '/random_sequences.fasta')
        self.hel.save_fasta(logdir + '/helical_sequences.fasta')


