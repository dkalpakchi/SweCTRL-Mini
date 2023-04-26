#!/bin/bash
to_process=([2786844]=1)
categories=("admin" "ads" "blogs" "debate" "forum" "info" "news" "review" "simple" "wiki" "lit")

for dir in $1/*/     # list directories in the form "/tmp/dirname/"
do
    mf=${dir%*/}      # remove the trailing "/"
    cid=`echo "$mf" | cut -d "-" -f 2`
 
    if [ -z "${to_process[$cid]}" ]; then
      continue
    fi

    echo $cid
    for category in "${categories[@]}"
    do
        python3 -m gen_hp_search.generate -c $category -l 255 -n 100 -o gen_hp_search/generated_$cid -f $mf -caf $1/ctrl_args.bin
    done
done
