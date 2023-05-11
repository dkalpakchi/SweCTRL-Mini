#!/bin/bash
datasets=("absaimm" "dalajged" "swediagnostics" "swefaq" "swefracas" "sweparaphrase" "swewic" "swewinogender" "swewinograd" "swedn" "swenli")

echo $1
for ((i = 0; i < ${#datasets[@]}; i++)) do
  echo ${datasets[$i]}
  python3 stats.py -d ${datasets[$i]} -s "$1"
  echo
done
 

