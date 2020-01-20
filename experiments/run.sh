#! /usr/bin/env bash

basedir=20200119
rddl=Reservoir-10
planner=hindsight

for dirpath in $basedir/$rddl/$planner/* ; do
    config=$dirpath/config.json

    if [ -f  "$config" ] ; then
        echo ">> [$(date +"%T")] Running tfplan for $config ..."
        cat $config
        echo
        echo
        echo ">> tfplan $planner $rddl --logdir $dirpath --config $config"
        tfplan $planner $rddl --logdir $dirpath --config $config
    fi
done
