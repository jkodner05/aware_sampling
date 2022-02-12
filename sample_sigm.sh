numsmalltrain=400
numlargetrain=8000
numtest=1000
numdev=1000
lt_loverlap=0.5
lt_foverlap=1.0
st_foverlap=0.5
indir=directory_that_contains_unimorphfiles_with_estimated_frequencies
outdir=outsigm

for lang in ara hye deu fin hun kan por rus zul
do
   python3 sample_sigm.py indir/"$lang"_wfreqs_smoothed.txt  $outdir --small $numsmalltrain --large $numlargetrain --dev $numdev --test $numtest --st_foverlap $st_foverlap --lt_foverlap $lt_foverlap --lt_loverlap $lt_loverlap --lang $lang --weighted

echo ""
done
