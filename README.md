# tf-plan [![Py Versions][py-versions.svg]][pypi-project] [![PyPI version][pypi-version.svg]][pypi-version] [![Build Status][travis.svg]][travis-project] [![Documentation Status][rtd-badge.svg]][rtd-badge] [![License: GPL v3][license.svg]][license]

Planning via gradient-based optimization in continuous MDPs using TensorFlow.

**tf-plan** is an implementation based on the NIPS 2017 paper:

> Wu Ga, Buser Say, and Scott Sanner, 2017<br>
> **[Scalable Planning with Tensorflow for Hybrid Nonlinear Domains.](http://papers.nips.cc/paper/7207-scalable-planning-with-tensorflow-for-hybrid-nonlinear-domains.pdf)**<br>
> In Advances in *Neural Information Processing Systems* (pp. 6273-6283).


# Quickstart

**tf-plan** is a Python3.5+ package available in PyPI.

```text
$ pip3 install -U tf-plan
```

# Features

**tf-plan** solves discrete time MDPs with continuous state-action spaces and deterministic transitions.

The domains/instances are specified using the [RDDL](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf) language.

It is built on Python3's RDDL toolkit:

- [pyrddl](https://github.com/thiagopbueno/pyrddl): RDDL lexer/parser in Python3.
- [rddl2tf](https://github.com/thiagopbueno/rddl2tf): RDDL2TensorFlow compiler.
- [rddlgym](https://github.com/thiagopbueno/rddlgym): A toolkit for working with RDDL domains in Python3.

Please refer to the projects' documentation for further details.


# Usage

```text
$ tfplan --help
Usage: tfplan [OPTIONS] [tensorplan|straightline|hindsight] RDDL

  Planning via gradient-based optimization in TensorFlow.

  Args:
      RDDL Filename or rddlgym domain/instance id.

Options:
  -b, --batch-size INTEGER        Number of trajectories in a batch.
                                  [default: 128]
  -hr, --horizon INTEGER          Number of timesteps.  [default: 40]
  -e, --epochs INTEGER            Number of training epochs.  [default: 500]
  --optimizer [Adadelta|Adagrad|Adam|GradientDescent|ProximalGradientDescent|ProximalAdagrad|RMSProp]
                                  [default: GradientDescent]
  -lr, --learning-rate FLOAT      Optimizer's learning rate.  [default: 0.001]
  -n, --num-samples INTEGER       Number of runs.  [default: 1]
  --num-workers INTEGER RANGE     Number of worker processes (min=1, max=12).
                                  [default: 1]
  --logdir PATH                   Directory used for logging training
                                  summaries.  [default: /tmp/tfplan/]
  -v, --verbose                   Verbosity flag.
  --version                       Show the version and exit.
  --help                          Show this message and exit.
```

## Examples

### Navigation

```text
$ tfplan tensorplan Navigation-v1 -b 32 -hr 20 -e 200 --optimizer RMSProp -lr 0.05

===== Average ===== (2.0904 ± 0.0000 sec)
       location(x)  location(y)    move(x)    move(y)  distance(z1)  distance(z2)  deceleration(z1)  deceleration(z2)     reward       done
count    20.000000    20.000000  20.000000  20.000000     20.000000     20.000000         20.000000         20.000000  20.000000  20.000000
mean      5.117020     5.964777   0.563600   0.587480      3.845318      5.233718          0.901469          0.898645  -4.208493   0.050000
std       2.734477     2.976508   0.457455   0.468573      1.745041      3.244598          0.141589          0.161664   4.017443   0.223607
min       0.000000     1.000000  -0.093972  -0.073001      1.136744      0.892700          0.574107          0.489666 -11.313708   0.000000
25%       3.102055     3.413130   0.021325   0.035666      2.156688      2.223654          0.844862          0.866897  -7.429872   0.000000
50%       5.206845     6.302579   0.822792   0.913129      4.295552      4.964723          0.985606          0.993987  -3.883022   0.000000
75%       7.966873     9.005674   0.984761   0.995023      5.404543      8.832995          0.996010          0.999950  -0.067122   0.000000
max       8.035465     9.124074   0.997835   0.996823      6.103278      8.956371          0.998212          0.999957  -0.007049   1.000000
```

# Documentation

Please refer to [https://tf-plan.readthedocs.io/](https://tf-plan.readthedocs.io/) for the code documentation.

# Support

If you are having issues with tf-plan, please let me know at: [thiago.pbueno@gmail.com](mailto://thiago.pbueno@gmail.com).

# License

Copyright (c) 2018-2020 Thiago Pereira Bueno All Rights Reserved.

tf-plan is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

tf-plan is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tf-plan. If not, see http://www.gnu.org/licenses/.


[py-versions.svg]: https://img.shields.io/pypi/pyversions/tf-plan.svg?logo=python&logoColor=white
[pypi-project]: https://pypi.org/project/tf-plan

[pypi-version.svg]: https://badge.fury.io/py/tf-plan.svg
[pypi-version]: https://badge.fury.io/py/tf-plan

[travis.svg]: https://img.shields.io/travis/thiagopbueno/tf-plan/master.svg?logo=travis
[travis-project]: https://travis-ci.org/thiagopbueno/tf-plan

[rtd-badge.svg]: https://readthedocs.org/projects/tf-plan/badge/?version=latest
[rtd-badge]: https://tf-plan.readthedocs.io/en/latest/?badge=latest

[license.svg]: https://img.shields.io/badge/License-GPL%20v3-blue.svg
[license]: https://github.com/thiagopbueno/tf-plan/blob/master/LICENSE
