#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

script="$0"
first="$1"
second="$2"

for i in {0..20..2}
do
#  if [ $1 == "mean" ]
#  then
#    python RAM_ASD_TD_mean_train_test.py --identifier mean --seed $i
#  elif [ $1 == "partial"]
#  then
#    python RAM_ASD_TD_partial_train_test.py --identifier partial --seed $i
#  else
#    python RAM_ASD_TD_partial_train_test.py --seed $i
#  fi
for d in {128,256,512}
do

#  if [$first = "no_attention" ]then
#    python RAM_ASD_TD_mean_train_test.py --latent $d --seed $i --model no_attention
#  elif [$first = "attention_only" ]
#  then
#    python RAM_ASD_TD_partial_train_test.py --latent $d --model attention_only --attention combine --seed $i
#  elif [ $second = "multiple"];
#  then
#    python RAM_ASD_TD_partial_train_test.py --latent $d --model combine --attention multiple --seed $i
#  elif [ $1 = "partial" ]
#  then
#    python RAM_ASD_TD_partial_train_test.py --latent $d --model combine --attention combine --seed $i
#  elif [ $1 = "partial" ]
#  then
#    python RAM_ASD_TD_partial_train_test.py --latent $d --model combine --attention sequential --seed $i
#  elif [ $1 = "partial" ]
#  then
#    python RAM_ASD_TD_partial_train_test.py --identifier partial --seed $i
#  fi
#
  if [[ $first = "no_attention" ]]; then
      python RAM_ASD_TD_partial_train_test.py --gpu 0 --latent $d --model no_attention --seed $i
  elif [[ $first = "attention_only" ]]; then
      python RAM_ASD_TD_partial_train_test.py --gpu 1 --latent $d --model attention_only --seed $i
  elif [[ $second = "multiple" ]]; then
      python RAM_ASD_TD_partial_train_test.py --gpu 2 --latent $d --model combine --attention multiple --seed $i
  elif [[ $second = "combine" ]]; then
      python RAM_ASD_TD_partial_train_test.py --gpu 3 --latent $d --model combine --attention combine --seed $i
  elif [[ $second = "sequential10" ]]; then
      python RAM_ASD_TD_partial_train_test.py --gpu 4 --latent $d --model combine --attention sequential --seed $i
  elif [[ $second = "sequential20" ]]; then
      python RAM_ASD_TD_partial_train_test.py --gpu 5 --latent $d --model combine --attention sequential --selen 20 --seed $i
  fi

done
done