import sys, os, argparse, random
from math import ceil
from numpy.random import choice, seed
from collections import defaultdict

random.seed("icgi2020")
seed(sum([ord(c) for c in "icgi2020"]))

LEMMA = 0
FEATS = 1
INFL = 2
FREQ = 3

def readcorpus(infname, weighted):
    # need to use indices because np.choice hates tuples and lists
    indices_to_words = {}
    umtriples_to_counts = {}
    with open(infname, "r") as fin:
        i = 0
        for line in fin:
            if not line.strip():
                continue
            if weighted:
                # requires frequences as fourth column
                lemma, infl, feats, freq = line.strip().split("\t")
                indices_to_words[i] = (lemma, infl, feats, float(freq))
            else:
                # ignores frequencies if present, no issue if absent
                lemma, infl, feats = line.strip().split("\t")[:3]
                indices_to_words[i] = (lemma, infl, feats, 1)
            i += 1
    return indices_to_words

def naive_sample(indices_to_words, numtrain, numdev, numtest):
    if numtrain + numtest > len(indices_to_words):
        print("Sampling too many!", numtrain+numtest, len(indices_to_words))
        exit()
    indices, words = zip(*tuple(indices_to_words.items()))
    sumcounts = sum([word[FREQ] for word in words])
    probs = [word[FREQ]/sumcounts for word in words]
    sample = choice(indices, numtrain+numdev+numtest, replace=False, p=probs)
    random.shuffle(sample) # need to shuffle since sample was weighted 
    trainsample = set(sample[:numtrain])
    devsample = set(sample[numtrain:numtrain+numdev])
    testsample = set(sample[numtrain+numdev:])
    return trainsample, devsample, testsample


def maxoverlap_sample(largetrainindices, indices_to_words, testindices, numtrain, maxoverlap):

    numtest = len(testindices)
    if numtrain + numtest > len(indices_to_words):
        print("Too many to sample!", numtrain+numtest, len(largetrainindices))
        exit()

    remainder = set(largetrainindices)
    #to get remainder, remove 1-maxoverlap items, most common feats first
    feats_to_test = defaultdict(lambda : list())
    for index in testindices:
        feat = indices_to_words[index][FEATS]
        feats_to_test[feat].append(index)

    limit = ceil((1-maxoverlap)*numtest)
    removablefeats = sorted(feats_to_test.items(), key=lambda kv : len(kv[1]), reverse = True)
    removedfeats = set()
    numremoved = 0
    for feat, items in removablefeats:
        if numremoved + len(items) < limit:
            numremoved += len(items)
            removedfeats.add(feat)
    for feat, items in reversed(removablefeats):
        if feat in removedfeats:
            continue
        numremoved += len(items)
        removedfeats.add(feat)
        if numremoved >= limit:
            break
    if numremoved < limit:
        print("Couldn't do it")

    remainingindices = [remaining for remaining in remainder if indices_to_words[remaining][FEATS] not in removedfeats]
    counts = [indices_to_words[remaining][FREQ] for remaining in remainingindices]
    sumcounts = sum(counts)
    probs = [count/sumcounts for count in counts]
    if len(remainingindices) < numtrain:
        print("Requested max overlap is too low to work!", len(remainingindices), numtrain)
        exit()
    trainsample = choice(remainingindices, numtrain, replace=False, p=probs)
    return trainsample
    


def compute_overlap(indices_to_words, train, test, i=FEATS, printoverlap=False):
    trainitems = [indices_to_words[index][i] for index in train]
    testitems = [indices_to_words[index][i] for index in test]
    numtestoverlap = 0
    numtrainoverlap = 0
    for item in testitems:
        if item in trainitems:
            numtestoverlap += 1
    for item in trainitems:
        if item in testitems:
            numtrainoverlap += 1
    if printoverlap:
        print("%overlap in train", 100*numtrainoverlap/len(trainitems))
        print("%overlap in test", 100*numtestoverlap/len(testitems))
    return 100*numtrainoverlap/len(trainitems), 100*numtestoverlap/len(testitems)



def multiple_sample(indices_to_words, largetrain, test, numsmalltrain, maxoverlap):
    currmaxoverlap = maxoverlap
    prevsmalltrain = None
    prevoverlap = 0
    # Progressively raise the the requested overlap
    #   in order to find the closest to requested that will work without going over
    #   Be willing tro try 100 times per request just in case of an unlucky sample
    for add in range(0,100-int(100*maxoverlap)):
        currmaxoverlap = maxoverlap + add/100.0
        for n in range(0,100):
            smalltrain = maxoverlap_sample(largetrain, indices_to_words, test, numsmalltrain, currmaxoverlap)
            smalltraino, smalltesto = compute_overlap(indices_to_words, smalltrain, test, i=FEATS)
            if smalltesto < maxoverlap*100:
                if smalltesto > prevoverlap:
                    prevsmalltrain = smalltrain
            elif smalltesto > maxoverlap and not prevsmalltrain is None:
                smalltrain = prevsmalltrain
                break
    return smalltrain



def writesample(indices_to_words, indices, showinfl, outdir, fname):
    outfname = os.path.join(outdir, fname)
    cutoff = 3
    if not showinfl:
        cutoff = 2
    with open(outfname, "w") as fout:
        for index in indices:
            word = indices_to_words[index][:cutoff]
            fout.write("%s\n" % "\t".join(word))


def main(infname, outdir, numsmalltrain, numlargetrain, numdev, numtest, maxoverlap, weighted, language):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Input:", infname)
    print("Output:", outdir)
    print("small train:", numsmalltrain, "large train:", numlargetrain, "dev:", numdev, "test:", numtest)
    print("Requested small test overlap", round(maxoverlap*100,2))

    indices_to_words = readcorpus(infname, weighted)

    # Feature overlap is consistently near 100% on large train, so nothing else
    #   needs to be done to ensure consistency
    # The test set and dev sets output here are reused for both train sizes
    largetrainindices, devindices, testindices = naive_sample(indices_to_words, numlargetrain, numdev, numtest)
    print("Achieved large-test overlap:")
    largetraino, largetesto = compute_overlap(indices_to_words, largetrainindices, testindices, i=FEATS, printoverlap=True)

    # Feature overlap needs to be controlled for for small train
    smalltrainindices = multiple_sample(indices_to_words, largetrainindices, testindices, numsmalltrain, maxoverlap)
    print("Achieved small-test overlap")
    smalltraino, smalltesto = compute_overlap(indices_to_words, smalltrainindices, testindices, i=FEATS, printoverlap=True)

    print("Achieved large-dev overlap:")
    largetraino, largedevo = compute_overlap(indices_to_words, largetrainindices, testindices, i=FEATS, printoverlap=True)
    print("Achieved dev-test overlap")
    devo, devtesto = compute_overlap(indices_to_words, devindices, testindices, i=1, printoverlap=True)

    print("Illicit large train triples in dev?", len(set(largetrainindices).intersection(set(devindices))))
    print("Illicit large train triples in test?", len(set(largetrainindices).intersection(set(testindices))))
    print("Illicit small train triples in dev?", len(set(smalltrainindices).intersection(set(devindices))))
    print("Illicit small train triples in test?", len(set(smalltrainindices).intersection(set(testindices))))
    print("Illicit dev triples in test?", len(set(smalltrainindices).intersection(set(testindices))))

    writesample(indices_to_words, testindices, False, outdir,  "%s_test.txt" % language)
    writesample(indices_to_words, devindices, True, outdir, "%s_dev.txt" % language)
    writesample(indices_to_words, smalltrainindices, True, outdir, "%s_train_small.txt" % language)
    writesample(indices_to_words, largetrainindices, True, outdir, "%s_train_large.txt" % language)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="make largetrain(smalltrain)/dev/test split with controlled smalltrain-test feature set overlap")
    parser.add_argument("infname", help="input file in UniMorph 3-column format, optionally with 4th frequency column")
    parser.add_argument("outdir", help="directory to write split files to")
    parser.add_argument("--small", type=int, help = "size of small training set")
    parser.add_argument("--large", type=int, help = "size of large training set")
    parser.add_argument("--dev", type=int, help = "size of dev set")
    parser.add_argument("--test", type=int, help = "size of test set")
    parser.add_argument("--maxoverlap", type=float, help = "requested max smalltrain-test feature set overlap. Float in range [0,1]")
    parser.add_argument("--weighted", action="store_true", help="weight sampling by frequency")
    parser.add_argument("--lang", help = "language to be written in output filenames")
    args = parser.parse_args()
    
    main(args.infname, args.outdir, args.small, args.large, args.dev, args.test, args.maxoverlap, args.weighted, args.lang)

