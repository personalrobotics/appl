#ifndef GLOBALS_H
#define GLOBALS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

#define MAP map
#define UMAP unordered_map
typedef std::pair<int, int> pii;

namespace Globals {

extern const double INF;
extern const double TINY;

struct Config {
  // Maximum depth of the search tree
  int search_depth;
  // Discount factor
  double discount;
  // Random-number seed
  unsigned int root_seed;
  // Amount of CPU time used for search during each move. Does not include the
  // time taken to prune the tree and update the belief.
  double time_per_move;
  // Number of starting states (samples)
  int n_particles;
  // Regularization parameter
  double pruning_constant;
  // Parameter such that eta * width(root) is the target uncertainty at the
  // root of the search tree, used in determining when to terminate a trial.
  double eta;
  // Number of moves to simulate
  int sim_len;
  // Whether the initial upper bound is approximate or true. If approximate,
  // the solver allows initial lower bound > initial upper bound at a node.
  bool approximate_ubound;

  Config() : 
    search_depth(90),
    discount(0.95),
    root_seed(42),
    time_per_move(1),
    n_particles(500),
    pruning_constant(0),
    eta(0.95),
    sim_len(-1),
    approximate_ubound(false)
  {}
};

extern Config config;

inline bool Fequals(double a, double b) { 
  return fabs(a - b) < TINY;
}

inline double ExcessUncertainty(
    double l, double u, double root_l, double root_u, int depth) {
  return (u-l) // width of current node
         - (config.eta * (root_u-root_l)) // epsilon
         * pow(config.discount, -depth);
}

inline void ValidateBounds(double& lb, double& ub) {
  if (ub >= lb)
    return;
  if (ub > lb - TINY || config.approximate_ubound)
    ub = lb;
  else
    assert(false);
}

template<class T>
inline void hash_combine(size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace

namespace std {
  template<typename S, typename T>
  struct hash<pair<S, T>> {
    inline size_t operator()(const pair<S, T>& v) const {
      size_t seed = 0;
      Globals::hash_combine(seed, v.first);
      Globals::hash_combine(seed, v.second);
      return seed;
    }
  };

  template<typename T>
  struct hash<vector<T>> {
    inline size_t operator()(const vector<T>& v) const {
      size_t seed = 0;
      for (const T& ele : v) {
        Globals::hash_combine(seed, ele);
      }
      return seed;
    }
  };
}

#endif
