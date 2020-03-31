#!/bin/bash

if [ "$1" == "" ]; then
  echo "Specify source as single positional parameter!"
  exit 1
fi

bash -c "exec -a 'MOJAVE polarization stacking --- ' python run_mojave_source.py $1"