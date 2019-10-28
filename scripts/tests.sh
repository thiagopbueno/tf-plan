#!/usr/bin/env bash

tfplan -v straightline Navigation-v3 -h 15 -e 1000 -lr 0.01 --optimizer GradientDescent --logdir experiments/nav-v3/straightline/sgd/lr=1e-2/epochs=1000
tfplan -v hindsight Navigation-v3 -h 15 -e 1000 -lr 0.01 --optimizer GradientDescent --logdir experiments/nav-v3/hindsight/sgd/lr=1e-2/epochs=1000

tfplan -v straightline Navigation-v3 -h 15 -e 1000 -lr 0.01 --optimizer Adam --logdir experiments/nav-v3/straightline/adam/lr=1e-2/epochs=1000
tfplan -v hindsight Navigation-v3 -h 15 -e 1000 -lr 0.01 --optimizer Adam --logdir experiments/nav-v3/hindsight/adam/lr=1e-2/epochs=1000

tfplan -v straightline Navigation-v3 -h 15 -e 1000 -lr 0.01 --optimizer RMSProp --logdir experiments/nav-v3/straightline/rmsprop/lr=1e-2/epochs=1000
tfplan -v hindsight Navigation-v3 -h 15 -e 1000 -lr 0.01 --optimizer RMSProp --logdir experiments/nav-v3/hindsight/rmsprop/lr=1e-2/epochs=1000
