#!/bin/bash
OUT_FNAME=ppl_lines.csv
rm $OUT_FNAME
for dir in $1/*/     # list directories in the form "/tmp/dirname/"
do
    mf=${dir%*/}      # remove the trailing "/"
    echo -n "$mf", >> $OUT_FNAME
    echo `python3 -m perplexity.main -d lines -s 8 -m "$mf" -caf $1/ctrl_args.bin -l -ov` >> $OUT_FNAME
done
