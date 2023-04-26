#!/bin/bash
to_process=([2786844]=1)
categories=("news/economy" "news/fashion" "news/food" "news/lifestyle" "news/opinion" "news/politics" "news/pressrelease" "news/science" "news/sport" "news/sustainability" "news/tech" "news/travel" "news/weather" "blogs/economy" "blogs/sport" "blogs/tech" "forum/economy" "forum/law" "forum/sport" "forum/tech" "forum/travel" "info/business" "info/lifestyle" "info/medical" "info/travel" "news/culture")

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
        python3 -m gen_hp_search.generate_minor -c $category -l 255 -n 100 -o gen_hp_search/generated_minor_v3_$cid -f $mf -caf $1/ctrl_args.bin
    done
done
