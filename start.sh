#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for i in {10..20..2}
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

  if [ $1 = "mean" ]
  then
    python RAM_ASD_TD_mean_train_test.py --identifier mean --seed $i
  elif [ $1 = "partial" ]
  then
    python RAM_ASD_TD_partial_train_test.py --identifier partial --seed $i
  else
    python RAM_ASD_TD_partial_train_test.py --identifier partial --seed $i
  fi

done