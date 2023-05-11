#!/bin/bash
declare -a datasets=("absa_imm" "dalaj" "swefracas" "swediagnostics" "swefaq" "swewic" "swewinogender" "swewinograd" "sweparaphrase")
NG="$1"

for ds in "${datasets[@]}"
do
	echo "$ds"
	python3 compute_swectrl_training_overlap.py -d "$ds" -s test -ng $NG -c es
done
