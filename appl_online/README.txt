                        ____  _____ ____  ____   ___ _____
                       |  _ \| ____/ ___||  _ \ / _ \_   _|
                       | | | |  _| \___ \| |_) | | | || |
                       | |_| | |___ ___) |  __/| |_| || |
                       |____/|_____|____/|_|    \___/ |_|


DESPOT is a C++ implementation of the DESPOT algorithm [1]. It takes as input
a POMDP model (coded in C++) and simulates an agent acting in the environment.

For bug reports and suggestions, please email <adhirajsomani@gmail.com>

[1] DESPOT: Online POMDP Planning with Regularization. Proc. Neural Information
Processing Systems (NIPS), 2013.


=================
TABLE OF CONTENTS
=================

* Requirements
* Quick Start
* Command Line Arguments
* Writing New Problems
* Package Contents

============
REQUIREMENTS
============

The build system is targeted to be used with `make` and a C++11 compiler.

Tested compilers:         gcc/g++ 4.8.1 under Linux
                          gcc/g++ 4.6.3 under Linux
                          clang++ 4.2   under Mac OS X Lion


===========
QUICK START
===========

After unzipping the package, cd into the extracted directory and run `make`
(see Makefile for compiler options). A single executable `despot` will be 
generated in the `bin` directory.

Parameters to the program are specified as command line arguments (see section
below). 6 sample problems are included in the package: Tag, LaserTag, 
RockSample, Tiger, Bridge and Pocman.

At each timestep the algorithm outputs the action, observation, reward and 
current state of the world. Additional information like the lower/upper bounds
before/after the search go to standard error and can be suppressed. The 
(un)discounted reward is output at the end of the run.

Examples:

./bin/despot -p tag -k 250 -d 80
  Run tag with 250 particles and search depth 80

./bin/despot -p rocksample -r 0.1 -b exact -s 7 -n 8
  Run rocksample(7, 8) with regularization parameter 0.1 and exact belief update

./bin/despot -p tiger -x 33 -l 100
  Run tiger with random-number seed 33 for upto 100 steps

./bin/despot -p pocman -g 1 -t 5 -a
  Run pocman with undiscounted search, 5 seconds per move, allowing the 
  initial lower bound at a node to be greater than its initial upper bound.

The following parameters give good results for the sample problems, using the
default search depth of 90 and discount of 0.95.

+------------+-----------+----------------+
| Problem    | Particles | Reg. Parameter |
+------------+-----------+----------------+
| Tag        | 500       | 0.01           |
| LaserTag   | 500       | 0.01           |
| RockSample | 500       | 0.1            |
| Tiger      | 50        | 0              |
| Bridge     | 100       | 0              |
| Pocman     | 500       | 0.01           |
+------------+-----------+----------------+


======================
COMMAND-LINE ARGUMENTS
======================

--help                 Print usage and exit.

-p <arg>               Problem name.
--problem=<arg>

-s <arg>               Size of the problem (problem-specific).
--size=<arg>

-n <arg>               Number of elements in the problem (problem-specific).
--number=<arg>

-d <arg>               Maximum depth of search tree (default: 90).
--depth=<arg>

-g <arg>               Discount factor for the search (default: 0.95).
--discount=<arg>

-x <arg>               Random number seed (default: 42).
--seed=<arg>

-t <arg>               Search time per move, in seconds (default: 1).
--timeout=<arg>

-k <arg>               Number of particles (default: 500).
--nparticles=<arg>

-r <arg>               Regularization parameter (default: no regularization).
--regparam=<arg>

-l <arg>               Number of steps to simulate (default: -1 = infinite).
--simlen=<arg>         Note that in some rare cases, when using a (relatively)
                       small number of particles, the program might run
                       infinitely without making progress (see design doc for
                       explanation). Therefore, it is advisable to limit the 
                       simulation length, especially in testing-scenarios where
                       many instances are run concurrently and a single process
                       shouldn't keep silently blocking forever.

-w <arg>               Lower bound strategy, if applicable. Can be either
--lbtype=<arg>         "mode" or "random". See src/Lower_bound/lower_bound_policy_mode.h
                       and src/lower_bound/lower_bound_policy_random.h for
                       more details.

-b <arg>               Belief-update strategy (default: 'particle'). Can be
--belief=<arg>         either "particle" or "exact". See src/belief_update/belief_update_particle.h
                       and src/belief_update/belief_update_exact.h for more
                       details.

-j <arg>               Knowledge level for random lower bound policy, if
--knowledge=<arg>      applicable. Level 1 generates legal actions, and level
                       2 generates preferred actions. See src/lower_bound/lower_bound_policy_random.h
                       for details about the 2 kinds of actions.

-a                     Whether the initial upper bound for a node is
--approx-upper-bound   approximate or true (default: false). 
                       If approximate, the solver allows the initial upper bound
                       to be small than the initial lower bound at a node,
                       bumping it up to the initial lower bound value. This may
                       be the case in complex problems like Pocman where it is 
                       hard to compute a true upper bound that is also 
                       sufficiently useful.


====================
WRITING NEW PROBLEMS
====================

1. Read the Overview section of `doc/Design.txt`.
2. Follow the guidelines in `src/model.h` to create new types for your
   problem that appropriately subclass the provided classes (see implementation
   of Tag for an example).
3. Put all your problem files in a subdirectory under `src/problems`. This
   enables the Makefile to pick them up automatically.
4. Modify main.cpp to recognize the new problem.
5. Compile and run.


================
PACKAGE CONTENTS
================

Makefile                       Build configuration
LICENSE                        License information
README                         This file
doc                            Documentation (Design, FAQ, Class diagram)
src/belief_update              Belief update strategies
src/globals                    Global data and functions
src/history                    Representation of an action-observation history
src/lower_bound                Implementation of lower bound functions
src/main                       Entry point for the program
src/memorypool                 Custom memory manager
src/model                      Model template (to be subclassed by new problems)
src/optionparser               Library for parsing command-line arguments
src/particle                   Representation of a particle
src/problems                   Directory containing problem implementations
src/qnode                      Representation of an action node (AND node)
src/random_streams             Representation of random-number sequences
src/solver                     Implementation of the algorithm
src/upper_bound                Implementation of upper bound functions
src/vnode                      Representation of a belief node
src/world                      Representation of the world (true state)

