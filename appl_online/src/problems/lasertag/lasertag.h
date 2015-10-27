#ifndef LASERTAG_H
#define LASERTAG_H

#include "globals.h"
#include "model.h"
#include "history.h"
#include "solver.h"
#include "lower_bound/lower_bound_policy_mode.h"
#include "upper_bound/upper_bound_stochastic.h"

class LaserTagState {
 public:
  LaserTagState() : id_(0) {}

  LaserTagState(int id) : id_(id) {}

  operator int() const { return id_; }

 private:
  int id_;
};

namespace std {
  template<>
  struct hash<LaserTagState> {
    inline size_t operator()(const LaserTagState& s) const {
      return s;
    }
  };
}

class LaserTag : public Model<LaserTagState> {
 public:
  LaserTag(unsigned initial_state_seed);

  void Step(LaserTagState& s, double random_num, int action, double& reward, 
            uint64_t& obs) const;

  void Step(LaserTagState& s, double random_num, int action, double& reward)
      const;

  double ObsProb(uint64_t z, const LaserTagState& s, int a) const;

  double FringeUpperBound(const LaserTagState& s) const;

  double FringeLowerBound(const vector<Particle<LaserTagState>*>& particles)
      const {
    return Globals::config.discount < 1 
           ? -1 / (1 - Globals::config.discount) 
           : -100;
  }

  int DefaultActionForState(const LaserTagState& s) const {
    return best_default_action_memo_[s]; 
  }

  LaserTagState GetStartState() const {
    unsigned seed = initial_state_seed_;
    return CellPairToState(rand_r(&seed) % n_cells_, rand_r(&seed) % n_cells_);
  }

  LaserTagState RandomState(unsigned& seed, uint64_t obs) const {
    return rand_r(&seed) % NumStates();
  }

  vector<pair<LaserTagState, double>> InitialBelief() const;

  void PrintState(const LaserTagState& state, ostream& out = cout) const;

  void PrintObs(uint64_t obs, ostream& out = cout) const {
    for (int i = 0; i < NBEAMS; i++) 
      out << GetReading(obs, i) << " ";
  }

  int NumStates() const { return n_states_; }

  int NumActions() const { return 5; }

  bool IsTerminal(const LaserTagState& s) const { 
    return s % (n_cells_ + 1) == n_cells_; 
  }

  uint64_t TerminalObs() const { return terminal_obs_; }

  Particle<LaserTagState>* Copy(const Particle<LaserTagState>* particle) const {
    Particle<LaserTagState>* new_particle = Allocate();
    *new_particle = *particle;
    return new_particle;
  }

  Particle<LaserTagState>* Allocate() const {
    return memory_pool_.Allocate();
  }

  void Free(Particle<LaserTagState>* particle) const {
    memory_pool_.Free(particle);
  }

  int DefaultActionPreferred(const History& history, const LaserTagState& state)
      const;

 private:
  bool Opposite(int d1, int d2) const {
    return (d1 == 0 && d2 == 1) || (d1 == 1 && d2 == 0) ||
           (d1 == 2 && d2 == 3) || (d1 == 3 && d2 == 2);
  }

  int TagAct() const { return NumActions() - 1; } 

  static int GetReading(uint64_t obs, uint64_t dir) {
    return (obs >> (dir * BITS_PER_READING)) & ((ONE << BITS_PER_READING) - 1);
  } 

  static void SetReading(uint64_t& obs, uint64_t reading, uint64_t dir) {
    // Clear bits
    obs &= ~( ((ONE << BITS_PER_READING) - 1) << (dir * BITS_PER_READING) );
    // Set bits
    obs |= reading << (dir * BITS_PER_READING);
  }

  // Returns the bucket that @noisy belongs to.
  // Buckets are numbered starting from 0 outwards from the robot. The size of
  // a bucket is given by @unit_size_. The last bucket may be smaller.
  int GetBucket(double noisy) const {
    return floor(noisy / unit_size_);
  }

  bool SameLocation(const LaserTagState& s) const {
    // Whether the robots are in the same cell in state s.
    return s % (n_cells_ + 2) == 0 || IsTerminal(s);
  }

  LaserTagState CellPairToState(int c1, int c2) const { 
    return c1 * (n_cells_ + 1) + c2;
  }

  bool WithinMap(int x, int y) const { 
    // Whether a cell is within the boundaries of the map and does not contain
    // an obstacle.
    return x >= 0 && x < floor_map_.size() && y >= 0 &&
        y < floor_map_[0].size() && floor_map_[x][y] != -1;
  }

  void InitReadingDistribution(int s);

  static double Erf(double x);

  static double Cdf(double x, double mean, double sigma);

  // Returns the set of cells, and their probabilities, that the opponent
  // can be in when the robot is in r_, the opponent is in o_, and an action
  // is taken. The opponent always tries to move away from the robot, so 
  // the actual action taken is irrelevant.
  UMAP<int, double> GetONextCells(int r_, int o_) const;

  // The cell occupied by the robot after taking action a in cell r
  int GetRNextCell(int r, int a) const;

  static constexpr uint64_t ONE = 1;
  static constexpr double TAG_REWARD = 10;
  static constexpr int BITS_PER_READING = 7;
  static constexpr int NBEAMS = 8;
  static uint64_t same_loc_obs_;
  static uint64_t terminal_obs_;

  // Seed to generate the starting state
  unsigned initial_state_seed_;

  // Map of the floor. -1 indicates obstacle, an integer indicates the cell
  // number. Cell numbers should be in range [0, number of free cells - 1],
  // but can be in any order in the map.
  vector<vector<int>> floor_map_;

  int rob_start_cell_;
  int opp_start_cell_;

  static constexpr double noise_sigma_ = 2.5;
  static constexpr double unit_size_ = 1.0;

  int n_cells_;
  int n_states_;

  // Mapping from cell number to (row, col)
  vector<pii> cell_to_coords_;

  // Mapping from state to pair of robot/opponent coordinates
  vector<pair<pii, pii>> state_to_coord_pair_;

  // Mapping from state to a pair of cell numbers occupied by the robot/opponent
  vector<pii> state_to_cell_pair_;

  // Transition model
  vector<vector<UMAP<LaserTagState, double>>> T;

  // Mapping from state to true distances to the nearest obstacle in each
  // direction
  vector<vector<double>> laser_dist_;

  // Table of best default actions for a given state
  vector<int> best_default_action_memo_;

  // readingDistributions[s][dir][reading] is the probability of @reading in
  // direction @dir when in state @s
  vector<vector<vector<double>>> reading_distributions_; 

  mutable MemoryPool<Particle<LaserTagState>> memory_pool_;
};

#endif
