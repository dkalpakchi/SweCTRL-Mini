#!/bin/bash
OUT_FNAME=ppl_swequad_mc.csv
declare -A processed=(
  [ctrl_minip_ddp_256_v4/checkpoint-1026732]=1 [ctrl_minip_ddp_256_v4/checkpoint-1075624]=1 [ctrl_minip_ddp_256_v4/checkpoint-1124516]=1 [ctrl_minip_ddp_256_v4/checkpoint-1173408]=1 [ctrl_minip_ddp_256_v4/checkpoint-1222300]=1 [ctrl_minip_ddp_256_v4/checkpoint-1271192]=1 [ctrl_minip_ddp_256_v4/checkpoint-1320084]=1 [ctrl_minip_ddp_256_v4/checkpoint-1368976]=1 [ctrl_minip_ddp_256_v4/checkpoint-1417868]=1 [ctrl_minip_ddp_256_v4/checkpoint-1466760]=1 [ctrl_minip_ddp_256_v4/checkpoint-1515652]=1 [ctrl_minip_ddp_256_v4/checkpoint-1564544]=1 [ctrl_minip_ddp_256_v4/checkpoint-1613436]=1 [ctrl_minip_ddp_256_v4/checkpoint-1662328]=1 [ctrl_minip_ddp_256_v4/checkpoint-1711220]=1 [ctrl_minip_ddp_256_v4/checkpoint-1760112]=1 [ctrl_minip_ddp_256_v4/checkpoint-1809004]=1 [ctrl_minip_ddp_256_v4/checkpoint-1857896]=1 [ctrl_minip_ddp_256_v4/checkpoint-1906788]=1 [ctrl_minip_ddp_256_v4/checkpoint-195568]=1 [ctrl_minip_ddp_256_v4/checkpoint-1955680]=1 [ctrl_minip_ddp_256_v4/checkpoint-2004572]=1 [ctrl_minip_ddp_256_v4/checkpoint-2053464]=1 [ctrl_minip_ddp_256_v4/checkpoint-2102356]=1 [ctrl_minip_ddp_256_v4/checkpoint-2151248]=1 [ctrl_minip_ddp_256_v4/checkpoint-2200140]=1 [ctrl_minip_ddp_256_v4/checkpoint-2249032]=1 [ctrl_minip_ddp_256_v4/checkpoint-2297924]=1 [ctrl_minip_ddp_256_v4/checkpoint-2346816]=1 [ctrl_minip_ddp_256_v4/checkpoint-2395708]=1 [ctrl_minip_ddp_256_v4/checkpoint-2444600]=1 [ctrl_minip_ddp_256_v4/checkpoint-2493492]=1 [ctrl_minip_ddp_256_v4/checkpoint-2542384]=1 [ctrl_minip_ddp_256_v4/checkpoint-2591276]=1
)

for dir in $1/*/     # list directories in the form "/tmp/dirname/"
do
    mf=${dir%*/}      # remove the trailing "/"
    if [ -n "${processed[$mf]}" ]; then
      continue
    fi
    echo -n "$mf", >> $OUT_FNAME
    echo `python3 -m perplexity.main -d swequad_mc -s 32 -m "$mf" -caf $1/ctrl_args.bin -l -ov` >> $OUT_FNAME
done
