#!/bin/bash
base_job_name="RAM_Sep_9"
job_file="the_job.sh"
identifier_name="RAM_attention"
dir="op"$identifier_name
mkdir -p $dir


for seed in {0,}; do
for latent in {256,}; do
  export seed latent
  export first="$1" second="$2"
  job_name=$base_job_name-$((first))-$((second))-$((seed))
  out_file=$dir/$base_job_name-$((seed)).out
  error_file=$dir/$base_job_name-$((seed)).err

  echo $seed $latent $first $second
  sbatch -J $job_name -o $out_file -t 1-00:00:00 -p tier3 -e $error_file $job_file
done
done
