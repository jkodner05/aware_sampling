# featbundle-aware_sampling

### For description of argments
```
$ python3 sample_sigm.py -h
```

### Sample execution. Unweighted sampling of German, up to 50% of feature sets that appear in test can appear in small training
```
$ python3 sample_sigm.py deu  deu_outdir --small 400 --large 4000 --dev 500 --test 500 --maxoverlap 0.5
lang "German" 
```
