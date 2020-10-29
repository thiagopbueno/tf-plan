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
    location(x)  location(y)   move(x)   move(y)  distance(z1)  distance(z2)  deceleration(z1)  deceleration(z2)     reward  done
0      0.000000     1.000000  0.997812  0.251873      6.103278      2.500000          0.998212          0.905148 -11.313708   0.0
1      0.901553     1.227575  0.997849  0.928313      5.244620      1.870730          0.995207          0.808415 -10.526089   0.0
2      1.704362     1.974440  0.997174  0.994867      4.152069      1.045723          0.983263          0.556283  -9.433639   0.0
3      2.249789     2.518605  0.991678  0.996339      3.389630      0.891024          0.960244          0.488901  -8.664492   0.0
4      2.715346     2.986350  0.891563  0.996716      2.740580      1.215423          0.917949          0.622608  -8.005720   0.0
5      3.224894     3.555996  0.421230  0.996869      2.010508      1.812289          0.819746          0.795918  -7.241465   0.0
6      3.499726     4.206403  0.444045  0.996535      1.528732      2.335447          0.705929          0.885625  -6.575032   0.0
7      3.777337     4.829425  0.924151  0.995804      1.266264      2.921140          0.621903          0.941680  -5.935030   0.0
8      4.318551     5.412601  0.984080  0.992210      1.138953      3.710104          0.574958          0.976961  -5.140282   0.0
9      4.871320     5.969937  0.985487  0.985184      1.475559      4.492920          0.690257          0.990931  -4.355447   0.0
10     5.545391     6.643800  0.975556  0.964462      2.212087      5.444489          0.854325          0.997096  -3.402467   0.0
11     6.376412     7.465371  0.940539  0.905565      3.269241      6.612030          0.954476          0.999284  -2.234082   0.0
12     7.273490     8.329092  0.739812  0.695562      4.453168      7.856998          0.988132          0.999839  -0.988905   0.0
13     8.004405     9.016289  0.029843  0.117345      5.424326      8.860193          0.996100          0.999952  -0.016874   0.0
14     8.034130     9.133170 -0.087813 -0.059488      5.538250      8.961620          0.996578          0.999957  -0.137474   0.0
15     7.946621     9.073888  0.021886 -0.044026      5.440866      8.857259          0.996173          0.999952  -0.091152   0.0
16     7.968422     9.030033  0.021429 -0.015050      5.415970      8.843177          0.996062          0.999951  -0.043579   0.0
17     7.989765     9.015043 -0.013826  0.016563      5.415193      8.848604          0.996059          0.999951  -0.018195   0.0
18     7.975994     9.031540 -0.028730  0.037703      5.421383      8.849745          0.996087          0.999951  -0.039636   0.0
19     7.947378     9.069093  0.010496 -0.030616      5.437246      8.854523          0.996157          0.999951  -0.086850   1.0
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
