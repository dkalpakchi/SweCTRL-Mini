#!/bin/bash
to_process=([2786844]=1)
categories=("news" "review" "news_sport" "blogs_sport" "news_travel" "info_travel" "wiki" "ads")

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
        python3 -m human_eval.generate -c $category -l 255 -n 3 -o human_eval/generated_$cid -f $mf -caf $1/ctrl_args.bin -pf human_eval/prompts_selected.yaml
    done
done
