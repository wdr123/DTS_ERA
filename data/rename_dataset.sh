#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..20..1}
do
  mv Fnl_Ds${i} dataset${i}
done