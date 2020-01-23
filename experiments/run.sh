#! /usr/bin/env bash

basedir=20200123

rddls=(Reservoir-10)
planners=(hindsight straightline)


run()
{
    total_num_experiments=$(find $1 -type f -name "config.json" -print | wc -l)

    i=0
    find $1 -type f -name "config.json" -print0 | while read -d $'\0' config
    do
        i=$((i + 1))
        echo "===== EXPERIMENT $i / $total_num_experiments ====="
        echo ">> tfplan $planner $rddl --logdir $(dirname $config) --config $config"
        echo ">> config = $(cat $config)"
        tfplan $planner $rddl --logdir $(dirname $config) --config $config
    done
}


for rddl in "${rddls[@]}"
do
    for planner in "${planners[@]}"
    do
        dirpath=$basedir/$rddl/$planner
        run $dirpath
    done
done
