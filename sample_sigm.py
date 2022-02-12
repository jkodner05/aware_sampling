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



def aware_sample(indices_to_words, numtrain, numdev, numtest, lemma_overlap_ratio=0.5, feat_overlap_ratio=1.0):

    def get_category_indices(indices_to_words, category):
        category_to_categoryindices = {}
        wordindices_to_categoryindices = {}
        cindex = -1
        for windex, wordtup in indices_to_words.items():
            cat = wordtup[category]
            if cat not in category_to_categoryindices:
                cindex += 1
                category_to_categoryindices[cat] = cindex
            wordindices_to_categoryindices[windex] = category_to_categoryindices[cat]
        return wordindices_to_categoryindices

    wordindices_to_lemmaindices = get_category_indices(indices_to_words, LEMMA)
    wordindices_to_featsindices = get_category_indices(indices_to_words, FEATS)

    indices, words = zip(*tuple(indices_to_words.items()))
    sumcounts = sum([word[FREQ] for i, word in enumerate(words)])
    probs = [word[FREQ]/sumcounts for word in words]
    # weighted sample w/o replacement
    sample = choice(indices, len(indices), replace=False, p=probs)
    # shuffle so that train/dev/test are drawn uniformly w/ respect to one another
    random.shuffle(sample)
    trainsample = set(sample[:numtrain])
    devsample = set(sample[numtrain:numtrain+numdev])

    # Now pick a prospective test set and swap items as necessary
    # to achieve desired percent overlaps
    testsample = sample[numtrain+numdev:numtrain+numdev+numtest]
    backuptest = sample[numtrain+numdev+numtest:]

    feats_in_train = set([wordindices_to_featsindices[i] for i in trainsample])
    lemmas_in_train = set([wordindices_to_lemmaindices[i] for i in trainsample])
    # Partition the backups into 4 generators to draw from as necessarily
    backup_fin_lin = (item for item in backuptest if wordindices_to_lemmaindices[item] in lemmas_in_train and wordindices_to_featsindices[item] in feats_in_train)
    backup_fin_lout = (item for item in backuptest if wordindices_to_lemmaindices[item] not in lemmas_in_train and wordindices_to_featsindices[item] in feats_in_train)
    backup_fout_lout = (item for item in backuptest if wordindices_to_lemmaindices[item] not in lemmas_in_train and wordindices_to_featsindices[item] not in feats_in_train)
    backup_fout_lin = (item for item in backuptest if wordindices_to_lemmaindices[item] in lemmas_in_train and wordindices_to_featsindices[item] not in feats_in_train)

    # Calculate target overlaps as counts
    desired_featoverlap = int(feat_overlap_ratio * numtest)
    current_featoverlap = len([item for item in testsample if wordindices_to_featsindices[item] in feats_in_train])
    desired_lemmaoverlap = int(lemma_overlap_ratio * numtest)
    current_lemmaoverlap = len([item for item in testsample if wordindices_to_lemmaindices[item] in lemmas_in_train])
    

    # Swap out items from the prospective test set with backup test items
    # Until the desired overlap ratios are reached or backups are exhausted.
    for i, item in enumerate((t for t in testsample)):
        if wordindices_to_featsindices[item] not in feats_in_train:
            if wordindices_to_lemmaindices[item] not in lemmas_in_train:
                newitem = None
                if current_featoverlap < desired_featoverlap: 
                    if current_lemmaoverlap < desired_lemmaoverlap:
                        newitem = next(backup_fin_lin, None) # attempt to increase lemma and feat overlap
                        current_featoverlap += 1 if newitem else 0
                        current_lemmaoverlap += 1 if newitem else 0
                    if current_lemmaoverlap >= desired_lemmaoverlap or newitem == None:
                        newitem = next(backup_fin_lout, None) # attempt to increase feat overlap, maintain lemma overlap
                        current_featoverlap += 1 if newitem else 0
                if current_featoverlap >= desired_featoverlap or newitem == None:
                    if current_lemmaoverlap <= desired_lemmaoverlap:
                        newitem = next(backup_fout_lin, None) # attempt to increase lemma overlap, maintain feat overlap
                        current_lemmaoverlap += 1 if newitem else 0
                    #if current_lemmaoverlap > desired_lemmaoverlap or newitem == None:
                    #    newitem = next(backup_fout_lout, None) # this case would do nothing
                if newitem:
                    testsample[i] = newitem
            elif wordindices_to_lemmaindices[item] in lemmas_in_train:
                newitem = None
                if current_featoverlap < desired_featoverlap:
                    if current_lemmaoverlap <= desired_lemmaoverlap:
                        newitem = next(backup_fin_lin, None) # attempt to increase feat overlap, maintain lemma overlap
                        current_featoverlap += 1 if newitem else 0
                    if current_lemmaoverlap > desired_lemmaoverlap or newitem == None:
                        newitem = next(backup_fin_lout, None) # attempt to increase feat overlap, decrease lemma overlap
                        current_featoverlap += 1 if newitem else 0
                        current_lemmaoverlap -= 1 if newitem else 0
                if current_featoverlap >= desired_featoverlap or newitem == None:
                    #if current_lemmaoverlap < desired_lemmaoverlap:
                    #    newitem = next(backup_fout_lin, None) # this case would do nothing
                    if current_lemmaoverlap > desired_lemmaoverlap or newitem == None:
                        newitem = next(backup_fout_lout, None) # attempt to maintain feat overlap, decrease lemma overlap
                        current_lemmaoverlap -= 1 if newitem else 0
                if newitem:
                    testsample[i] = newitem
        elif wordindices_to_featsindices[item] in feats_in_train:
            if wordindices_to_lemmaindices[item] not in lemmas_in_train:
                newitem = None
                if current_featoverlap <= desired_featoverlap:
                    if current_lemmaoverlap < desired_lemmaoverlap:
                        newitem = next(backup_fin_lin, None) # attempt to maintain feat overlap, increase lemma overlap
                        current_lemmaoverlap += 1 if newitem else 0
                    #if current_lemmaoverlap >= desired_lemmaoverlap or newitem == None:
                    #    newitem = next(backup_fin_lout, None) # this case would do nothing
                if current_featoverlap > desired_featoverlap or newitem == None:
                    if current_lemmaoverlap < desired_lemmaoverlap:
                        newitem = next(backup_fout_lin, None) # attempt to lower feat overlap, increase lemma overlap
                        current_featoverlap -= 1 if newitem else 0
                        current_lemmaoverlap += 1 if newitem else 0
                    if current_lemmaoverlap >= desired_lemmaoverlap or newitem == None:
                        newitem = next(backup_fout_lout, None) # attempt o decreae feat overlap, maintain lemma overlap
                        current_featoverlap -= 1 if newitem else 0
                if newitem:
                    testsample[i] = newitem
            elif wordindices_to_lemmaindices[item] in lemmas_in_train:
                newitem = None
                if current_featoverlap <= desired_featoverlap:
                    #if current_lemmaoverlap <= desired_lemmaoverlap:
                    #    newitem = next(backup_fin_lin, None) # this case would do nothing
                    if current_lemmaoverlap > desired_lemmaoverlap:
                        newitem = next(backup_fin_lout, None) # attempt to maintain feat overlap, decrease lemma overlap
                        current_lemmaoverlap -= 1 if newitem else 0
                if current_featoverlap > desired_featoverlap or newitem == None:
                    if current_lemmaoverlap <= desired_lemmaoverlap:
                        newitem = next(backup_fout_lin, None) # attempt to decrease feat overlap, maintain lemma overlap
                        current_featoverlap -= 1 if newitem else 0
                    if current_lemmaoverlap > desired_lemmaoverlap or newitem == None:
                        newitem = next(backup_fout_lout, None) # attempt to decrease both feat overlap and lemma overlap
                        current_lemmaoverlap -= 1 if newitem else 0
                        current_featoverlap -= 1 if newitem else 0
                if newitem:
                    testsample[i] = newitem
        if current_featoverlap == desired_featoverlap and current_lemmaoverlap == desired_lemmaoverlap:
            break # yay, we did it!
        # If we didn't succeed perfectly, we got as close as the test sample will allow
        # The other option would be to resample and try again, hoping we get a better test sample

    return trainsample, devsample, testsample



def naive_sample(indices_to_words, numtrain, numdev, numtest):
    if numtrain + numtest > len(indices_to_words):
        print("Sampling too many!", numtrain+numtest, len(indices_to_words))
        exit()
    indices, words = zip(*tuple(indices_to_words.items()))
    sumcounts = sum([word[FREQ] for word in words])
    probs = [word[FREQ]/sumcounts for word in words]
    # weighted sample w/o replacement
    sample = choice(indices, numtrain+numdev+numtest, replace=False, p=probs)
    # shuffle so that train/dev/test are drawn uniformly w/ respect to one another
    random.shuffle(sample) # need to shuffle since sample was weighted 
    trainsample = set(sample[:numtrain])
    devsample = set(sample[numtrain:numtrain+numdev])
    testsample = set(sample[numtrain+numdev:])
    return trainsample, devsample, testsample


def maxoverlap_sample(largetrainindices, indices_to_words, testindices, numtrain, max_foverlap):

    # The test set is already fixed at this point, so the only option is to manipulate the small training set
    # This is tricker because changing one item from the small training set can change the test overlap by more than one
    
    numtest = len(testindices)
    if numtrain + numtest > len(indices_to_words):
        print("Too many to sample!", numtrain+numtest, len(largetrainindices))
        exit()

    remainder = set(largetrainindices)
    #to get remainder, remove 1-max_foverlap items, most common feats first
    feats_to_test = defaultdict(lambda : list())
    for index in testindices:
        feat = indices_to_words[index][FEATS]
        feats_to_test[feat].append(index)

    limit = ceil((1-max_foverlap)*numtest)
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


def main(infname, outdir, numsmalltrain, numlargetrain, numdev, numtest, lt_loverlap, lt_foverlap, st_foverlap, weighted, language):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Input:", infname)
    indices_to_words = readcorpus(infname, weighted)
    print("contains %s items" % len(indices_to_words))
    print("Output:", outdir)
    print("small train:", numsmalltrain, "large train:", numlargetrain, "dev:", numdev, "test:", numtest)
    print("Requested large test lemma overlap", round(lt_loverlap*100,2))
    print("Requested large test feature overlap", round(lt_foverlap*100,2))
    print("Requested small test feature overlap", round(st_foverlap*100,2))

    if len(indices_to_words) >= numlargetrain + numsmalltrain + numdev + numtest:
        # Feature overlap is consistently near 100% on large train, so nothing else
        #   needs to be done to ensure consistency
        # The test set and dev sets output here are reused for both train sizes
        if lt_loverlap and lt_foverlap:
            largetrainindices, devindices, testindices = aware_sample(indices_to_words, numlargetrain, numdev, numtest, lemma_overlap_ratio=lt_loverlap, feat_overlap_ratio=lt_foverlap)
        else:
            largetrainindices, devindices, testindices = naive_sample(indices_to_words, numlargetrain, numdev, numtest)

        print("Achieved large-test feature overlap:")
        largetraino, largetesto = compute_overlap(indices_to_words, largetrainindices, testindices, i=FEATS, printoverlap=True)
        print("Achieved large-test lemma overlap:")
        largetraino, largetesto = compute_overlap(indices_to_words, largetrainindices, testindices, i=LEMMA, printoverlap=True)

        smalltrainindices = multiple_sample(indices_to_words, largetrainindices, testindices, numsmalltrain, st_foverlap)
        print("Achieved small-test feature overlap")
        smalltraino, smalltesto = compute_overlap(indices_to_words, smalltrainindices, testindices, i=FEATS, printoverlap=True)
        print("Achieved small-test lemma overlap")
        smalltraino, smalltesto = compute_overlap(indices_to_words, smalltrainindices, testindices, i=LEMMA, printoverlap=True)

        print("Illicit large train triples in dev?", len(set(largetrainindices).intersection(set(devindices))))
        print("Illicit large train triples in test?", len(set(largetrainindices).intersection(set(testindices))))

        writesample(indices_to_words, largetrainindices, True, outdir, "%s_train_large.txt" % language)

    elif len(indices_to_words) >= numsmalltrain+numdev+numtest:
        # If there isn't enough data for large train, directly sample small train
        print("TOO SMALL TO GENERATE LARGE TRAINING SET. Generating small training directly")
        if lt_loverlap:
            smalltrainindices, devindices, testindices = aware_sample(indices_to_words, numsmalltrain, numdev, numtest, lemma_overlap_ratio=lt_loverlap, feat_overlap_ratio=st_foverlap)
        else:
            smalltrainindices, devindices, testindices = naive_sample(indices_to_words, numsmalltrain, numdev, numtest)
        print("Achieved small-test feature overlap")
        smalltraino, smalltesto = compute_overlap(indices_to_words, smalltrainindices, testindices, i=FEATS, printoverlap=True)
        print("Achieved small-test lemma overlap")
        smalltraino, smalltesto = compute_overlap(indices_to_words, smalltrainindices, testindices, i=LEMMA, printoverlap=True)

    else:
        print("DATA SET TOO SMALL!")
        exit()

    print("Illicit small train triples in dev?", len(set(smalltrainindices).intersection(set(devindices))))
    print("Illicit small train triples in test?", len(set(smalltrainindices).intersection(set(testindices))))
    print("Illicit dev triples in test?", len(set(smalltrainindices).intersection(set(testindices))))

    writesample(indices_to_words, testindices, False, outdir,  "%s_test.txt" % language)
    writesample(indices_to_words, testindices, True, outdir,  "%s_testgold.txt" % language)
    writesample(indices_to_words, devindices, True, outdir, "%s_dev.txt" % language)
    writesample(indices_to_words, smalltrainindices, True, outdir, "%s_train_small.txt" % language)




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="make largetrain(smalltrain)/dev/test split with controlled smalltrain-test feature set overlap")
    parser.add_argument("infname", help="input file in UniMorph 3-column format, optionally with 4th frequency column")
    parser.add_argument("outdir", help="directory to write split files to")
    parser.add_argument("--small", type=int, help = "size of small training set")
    parser.add_argument("--large", type=int, help = "size of large training set")
    parser.add_argument("--dev", type=int, help = "size of dev set")
    parser.add_argument("--test", type=int, help = "size of test set")
    parser.add_argument("--lt_loverlap",  type=float, help = "requested max largetrain-test lemma set overlap. Float in range [0,1]. Optional. If this or lt_floverlap are omitted, overlaps are uncontrolled in large train")
    parser.add_argument("--lt_foverlap",  type=float, help = "requested max largetrain-test feature set overlap. Float in range [0,1]. Optional. If this or lt_lloverlap are omitted, overlaps are uncontrolled in large train")
    parser.add_argument("--st_foverlap", type=float, help = "requested max smalltrain-test feature set overlap. Float in range [0,1]")
    parser.add_argument("--weighted", action="store_true", help="weight sampling by frequencies provided in 4th column")
    parser.add_argument("--lang", help = "language to be written in output filenames")
    args = parser.parse_args()
    
    main(args.infname, args.outdir, args.small, args.large, args.dev, args.test, args.lt_loverlap, args.lt_foverlap, args.st_foverlap, args.weighted, args.lang)

