#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..20..1}
do
  python RAM_ASD_TD_mean_train_test.py --identifier mean --seed $i
done