#ifndef TAG_H
#define TAG_H

#include "globals.h"
#include "model.h"
#include "solver.h"
#include "history.h"
#include "lower_bound/lower_bound_policy_mode.h"
#include "lower_bound/lower_bound_policy_random.h"
#include "upper_bound/upper_bound_stochastic.h"

class TagState {
 public:
  TagState() : id_(0) {}

  TagState(int id) : id_(id) {}

  operator int() const { return id_; }

 private:
  int id_;
};

namespace std {
  template<>
  struct hash<TagState> {
    inline size_t operator()(const TagState& s) const {
      return s;
    }
  };
}

class Tag : public Model<TagState> {
 public:
  Tag(unsigned initial_state_seed); 
  
  void Step(TagState& s, double random_num, int action, double& reward,
            uint64_t& obs) const;

  void Step(TagState& s, double random_num, int action, double& reward) const;

  double ObsProb(uint64_t z, const TagState& s, int action) const;

  double FringeUpperBound(const TagState& s) const;

  double FringeLowerBound(const vector<Particle<TagState>*>& particles) const {
    return Globals::config.discount < 1 
           ? -1 / (1 - Globals::config.discount) 
           : -100;
  }

  int DefaultActionForState(const TagState& s) const {
    return best_default_action_memo_[s]; 
  }

  TagState GetStartState() const {
    return CellPairToState(rob_start_cell_, opp_start_cell_);
  }

  TagState RandomState(unsigned& seed, uint64_t obs) const {
    return rand_r(&seed) % NumStates();
  }

  vector<pair<TagState, double>> InitialBelief() const;

  void PrintState(const TagState& s, ostream& out = cout) const; 

  void PrintObs(uint64_t obs, ostream& out = cout) const { out << obs; }

  int NumStates() const { return n_states_; }

  int NumActions() const { return 5; }

  bool IsTerminal(const TagState& s) const { 
    return s % (n_cells_ + 1) == n_cells_; 
  }

  uint64_t TerminalObs() const { return n_cells_ + 1; }

  Particle<TagState>* Copy(const Particle<TagState>* particle) const {
    Particle<TagState>* new_particle = Allocate();
    *new_particle = *particle;
    return new_particle;
  }

  Particle<TagState>* Allocate() const {
    return memory_pool_.Allocate();
  }

  void Free(Particle<TagState>* particle) const {
    memory_pool_.Free(particle);
  }

  int DefaultActionPreferred(const History& history, const TagState& state) 
      const;

 private:
  bool Opposite(int d1, int d2) const {
    return (d1 == 0 && d2 == 1) || (d1 == 1 && d2 == 0) ||
           (d1 == 2 && d2 == 3) || (d1 == 3 && d2 == 2);
  }

  int TagAct() const { return NumActions() - 1; }

  bool SameLocation(const TagState& s) const {
    // Whether the robots are in the same cell in state s.
    return s % (n_cells_ + 2) == 0 || IsTerminal(s);
  }

  TagState CellPairToState(int c1, int c2) const { 
    return c1 * (n_cells_ + 1) + c2;
  }

  bool WithinMap(int x, int y) const { 
    // Whether a cell is within the boundaries of the map and does not contain
    // an obstacle.
    return x >= 0 && x < floor_map_.size() && y >= 0 &&
        y < floor_map_[0].size() && floor_map_[x][y] != -1;
  }

  // Returns the set of cells, and their probabilities, that the opponent
  // can be in when the robot is in r_, the opponent is in o_, and an action
  // is taken. The opponent always tries to move away from the robot, so 
  // the actual action taken is irrelevant.
  UMAP<int, double> GetONextCells(int r_, int o_) const;

  // The cell occupied by the robot after taking action a in cell r
  int GetRNextCell(int r, int a) const;

  static constexpr double TAG_REWARD = 10;

  // Map of the floor. -1 indicates an obstacle, an integer indicates the cell
  // number. Cell numbers should be in range [0, number of free cells - 1],
  // but can be in any order in the map.
  vector<vector<int>> floor_map_;

  int rob_start_cell_;
  int opp_start_cell_;

  int n_cells_;
  int n_states_;

  // Mapping from cell number to (row, col)
  vector<pii> cell_to_coords_;

  // Mapping from state to pair of robot/opponent coordinates
  vector<pair<pii, pii>> state_to_coord_pair_;

  // Mapping from state to a pair of cell numbers occupied by the robot/opponent
  vector<pii> state_to_cell_pair_;

  // Transition model
  vector<vector<UMAP<TagState, double>>> T_;

  // Table of best default actions for a given state
  vector<int> best_default_action_memo_;

  mutable MemoryPool<Particle<TagState>> memory_pool_;
};

#endif
