#!/bin/bash -lT


#SBATCH --mail-user=986739772@qq.com
#SBATCH --mail-type=ALL
#SBATCH -A vision -p tier3 -n 2
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH --gres=gpu:a100:1

conda activate ADS
if [[ $first = "no_attention" ]]; then
      python RAM_ASD_TD_train_test.py --latent $latent --model no_attention --seed $seed
  elif [[ $first = "attention_only" ]]; then
      python RAM_ASD_TD_train_test.py --latent $latent --model attention_only --seed $seed
  elif [[ $second = "multiple" ]]; then
      python RAM_ASD_TD_train_test.py --latent $latent --model combine --attention multiple --seed $seed
  elif [[ $second = "combine" ]]; then
      python RAM_ASD_TD_train_test.py --latent $latent --model combine --attention combine --seed $seed
  elif [[ $second = "sequential10" ]]; then
        python RAM_ASD_TD_train_test.py --latent $latent --model combine --attention sequential --seed $seed
  elif [[ $second = "sequential20" ]]; then
        python RAM_ASD_TD_train_test.py --latent $latent --model combine --attention sequential --selen 20 --seed $seed
fi

