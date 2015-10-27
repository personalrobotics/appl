#ifndef ROCKSAMPLE_H 
#define ROCKSAMPLE_H 

#include "model.h"
#include "lower_bound/lower_bound.h"
#include "upper_bound/upper_bound_nonstochastic.h"

class RockSampleState {
 public:
  RockSampleState() : id_(0) {}

  RockSampleState(int id) : id_(id) {}

  operator int() const { return id_; }

 private:
  int id_;
};

namespace std {
  template<>
  struct hash<RockSampleState> {
    inline size_t operator()(const RockSampleState& s) const {
      return s;
    }
  };
}

/* Note: RockSample inherits directly from ILowerBound to provide a custom
 * lower bound.
 */
class RockSample : public Model<RockSampleState>,
                   public ILowerBound<RockSampleState> {
 public:
  RockSample(int size, int rocks, unsigned initial_state_seed,
             const RandomStreams& streams);

  void Step(RockSampleState& s, double random_num, int action, double& reward, 
            uint64_t& obs) const {
    reward = R[s][action];
    if (action < 5)
      obs = IsTerminal(T[s][action]) ? TerminalObs() : kNone;
    else {
      int rock_cell = rocks_[action - 5];
      int agent_cell = CellOf(s);
      double eff = eff_[agent_cell][rock_cell];
      obs = (random_num <= eff) == RockStatus(action - 5, s);
    }
    s = T[s][action];
  }

  void Step(RockSampleState& s, double random_num, int action, double& reward)
      const {
    reward = R[s][action];
    s = T[s][action];
  }

  double FringeUpperBound(const RockSampleState& s) const;
  
  double ObsProb(uint64_t z, const RockSampleState& s, int action) const;

  RockSampleState GetStartState() const {
    return MakeState(robot_start_cell_, rock_set_start_);
  }

  RockSampleState RandomState(unsigned& seed, uint64_t obs) const {
    return rand_r(&seed) % NumStates();
  }

  vector<pair<RockSampleState, double>> InitialBelief() const;

  void PrintState(const RockSampleState& state, ostream& out = cout) const;

  void PrintObs(uint64_t obs, ostream& out = cout) const {
    switch(obs) {
      case kNone: out << "NONE"; break;
      case kGood: out << "GOOD"; break;
      case kBad: out << "BAD"; break;
      case kTerminal: out << "TERMINAL"; break;
      default: out << "UNKNOWN";
    }
  }

  int NumActions() const { return K_ + 5; }

  int NumStates() const { return n_states_; }

  bool IsTerminal(const RockSampleState& s) const { 
    return CellOf(s) == n_cells_; 
  }

  uint64_t TerminalObs() const { return kTerminal; }

  vector<vector<UMAP<RockSampleState, double>>> TransitionMatrix() const;

  pair<double, int> LowerBound(
      History& history,
      const vector<Particle<RockSampleState>*>& particles,
      int stream_position,
      const Model<RockSampleState>& model) const;

  Particle<RockSampleState>* Copy(const Particle<RockSampleState>* particle)
      const {
    Particle<RockSampleState>* new_particle = Allocate();
    *new_particle = *particle;
    return new_particle;
  }

  Particle<RockSampleState>* Allocate() const {
    return memory_pool_.Allocate();
  }

  void Free(Particle<RockSampleState>* particle) const {
    memory_pool_.Free(particle);
  }

  // A pointer to the upper bound actions as computed in 
  // UpperBoundNonStochastic; used in the custom lower bound
  void set_upper_bound_act(const vector<int>& upper_bound_act) {
    upper_bound_act_ = &upper_bound_act;
  }

 private:
  void Init_7_8();
  void Init_11_11();
  void InitGeneral(unsigned& seed);

  int CellNum(int row, int col) const {
    return row * size_ + col;
  }

  RockSampleState MakeState(int cell, int rock_set) const {
    return (cell << K_) + rock_set;
  }

  // True for good rock, false for bad rock
  // x can be a rock set or State
  int RockStatus(int rock, int x) const {
    return (x >> rock) & 1;
  }

  // The rock set after sampling a rock from it
  int Sample(int rock, int rock_set) const {
    return rock_set & ~(1 << rock);
  }

  // The set of rocks in the state
  int RockSetOf(const RockSampleState& s) const {
    return s & ((1<<K_)-1);
  }

  // Which cell the agent is in
  int CellOf(const RockSampleState& s) const {
    return s >> K_;
  }

  static constexpr double half_eff_distance_ = 20;

  enum {
    kBad, // Must always be 0 for implicit conversion from RockStatus()
    kGood, // Must always be 1 for implicit conversion from RockStatus()
    kNone,
    kTerminal
  };
  vector<pii> cell_to_coords_; // Mapping from cell number to (row, col)
  vector<int> rocks_; // Mapping from rock # -> cell number
  int size_;
  int robot_start_cell_; // Starting cell of the agent
  int rock_set_start_; // Starting rock configuration
  int n_cells_; // # of cells excluding terminal cell
  int n_states_; // # of states excluding terminal state
  int K_; // # of rocks
  // Mapping from cell number to rock number (-1 if there's no rock in the cell)
  vector<int> rock_at_cell_;
  vector<vector<RockSampleState>> T; // Transition matrix
  vector<vector<RockSampleState>> R; // Reward matrix
  vector<vector<double>> eff_; // Precomputed efficiency for each pair of cells

  // Helper tables used in computing the lower bound
  mutable vector<double> weight_sum_of_state_;
  mutable vector<RockSampleState> state_seen_;

  const vector<int>* upper_bound_act_;

  mutable MemoryPool<Particle<RockSampleState>> memory_pool_;
};

#endif
