#include "rocksample.h"

void RockSample::Init_7_8() {
  rocks_ = {
    CellNum(0, 1),
    CellNum(1, 5),
    CellNum(2, 2),
    CellNum(2, 3),
    CellNum(3, 6),
    CellNum(5, 0),
    CellNum(5, 3),
    CellNum(6, 2),
  };
  robot_start_cell_ = CellNum(3, 0);
}

void RockSample::Init_11_11() {
  rocks_ = {
    CellNum(7, 0),
    CellNum(3, 0),
    CellNum(2, 1),
    CellNum(6, 2),
    CellNum(7, 3),
    CellNum(2, 3),
    CellNum(7, 4),
    CellNum(2, 5),
    CellNum(9, 6),
    CellNum(7, 9),
    CellNum(1, 9)
  };
  robot_start_cell_ = CellNum(5, 0);
}

void RockSample::InitGeneral(unsigned& seed) {
  do {
    int cell = rand_r(&seed) % (size_ * size_);
    if (find(rocks_.begin(), rocks_.end(), cell) == rocks_.end())
      rocks_.push_back(cell);
  } while (rocks_.size() < K_);
  robot_start_cell_ = CellNum(size_ / 2, 0);
}

RockSample::RockSample(int size, int rocks, unsigned initial_state_seed, 
                  const RandomStreams& streams)
    : ILowerBound<RockSampleState>(streams),
      size_(size),
      K_(rocks) {
  if (size == 7 && rocks == 8)
    Init_7_8();
  else if (size == 11 && rocks == 11)
    Init_11_11();
  else
    InitGeneral(initial_state_seed);

  rock_set_start_ = 0;
  for (int i = 0; i < rocks_.size(); i++)
    if (rand_r(&initial_state_seed) & 1)
      rock_set_start_ |= (1 << i);

  n_cells_ = size_ * size_;

  rock_at_cell_.resize(n_cells_, -1);
  for (int i = 0; i < K_; i++)
    rock_at_cell_[rocks_[i]] = i;
  
  cell_to_coords_.resize(n_cells_);
  for (int i = 0; i < n_cells_; i++)
    cell_to_coords_[i] = {i / size_, i % size_};

  n_states_ = (n_cells_ + 1) * (1 << K_); // +1 for exit location

  weight_sum_of_state_.resize(n_states_);

  state_seen_.resize(n_states_);

  // T and R
  T.resize(n_states_);
  R.resize(n_states_);
  for (int cell = 0; cell < n_cells_; cell++) {
    for (int rock_set = 0; rock_set < (1 << K_); rock_set++) {
      RockSampleState s = MakeState(cell, rock_set);
      T[s].resize(NumActions());
      R[s].resize(NumActions());
      pii ac = cell_to_coords_[cell];
      // North
      if (ac.first == 0) {
        T[s][0] = s;
        R[s][0] = -100;
      }
      else {
        T[s][0] = MakeState(CellNum(ac.first-1, ac.second), rock_set);
        R[s][0] = 0;
      }
      // South
      if (ac.first == size_-1) {
        T[s][1] = s;
        R[s][1] = -100;
      }
      else {
        T[s][1] = MakeState(CellNum(ac.first+1, ac.second), rock_set);
        R[s][1] = 0;
      }
      // East
      if (ac.second == size_-1) {
        T[s][2] = MakeState(n_cells_, rock_set);
        R[s][2] = +10;
      }
      else {
        T[s][2] = MakeState(CellNum(ac.first, ac.second+1), rock_set);
        R[s][2] = 0;
      }
      // West
      if (ac.second == 0) {
        T[s][3] = s;
        R[s][3] = -100;
      }
      else {
        T[s][3] = MakeState(CellNum(ac.first, ac.second-1), rock_set);
        R[s][3] = 0;
      }
      // Sample
      int rock = rock_at_cell_[cell];
      if (rock != -1) {
        if (RockStatus(rock, rock_set)) {
          T[s][4] = MakeState(cell, Sample(rock, rock_set));
          R[s][4] = +10;
        }
        else {
          T[s][4] = s;
          R[s][4] = -10;
        }
      }
      else {
        T[s][4] = s;
        R[s][4] = -100;
      }
      // Check
      for (int a = 5; a < NumActions(); a++) {
        T[s][a] = s;
        R[s][a] = 0;
      }
    }
  }
  // Terminal states
  for (int k = 0; k < (1 << K_); k++) {
    RockSampleState s = MakeState(n_cells_, k);
    T[s].resize(NumActions());
    R[s].resize(NumActions());
    for (int a = 0; a < NumActions(); a++) {
      T[s][a] = s;
      R[s][a] = 0;
    }
  }

  // eff_
  eff_.resize(n_cells_);
  for (int i = 0; i < n_cells_; i++) {
    eff_[i].resize(n_cells_);
    for (int j = 0; j < n_cells_; j++) {
      pii agent = cell_to_coords_[i];
      pii other = cell_to_coords_[j];
      double dist = sqrt(pow(agent.first-other.first, 2) +
                         pow(agent.second-other.second, 2));
      eff_[i][j] = (1 + pow(2, -dist / half_eff_distance_)) * 0.5;
    }
  }
}

double RockSample::ObsProb(uint64_t obs, const RockSampleState& s, int a) 
    const {
  // Terminal state should match terminal obs
  if (IsTerminal(s))
    return obs == TerminalObs();

  if (a < 5)
    return obs == kNone;

  if (obs != kGood && obs != kBad)
    return 0;

  int rock = a - 5;
  int rock_cell = rocks_[rock];
  int agent_cell = CellOf(s);
  double eff = eff_[agent_cell][rock_cell];
  if (obs == RockStatus(rock, s))
    return eff;

  return 1 - eff;
}

double RockSample::FringeUpperBound(const RockSampleState& s) const {
  if (IsTerminal(s))
    return 0; 

  int rock_set = RockSetOf(s);
  int n_good = 0;
  while (rock_set) {
    n_good += rock_set & 1;
    rock_set >>= 1;
  }

  // Assume a good rock is sampled at each step and an exit is made in the last
  if (Globals::config.discount < 1)
    return 10 * (1 - pow(Globals::config.discount, (n_good+1))) / (1 - Globals::config.discount);
  else
    return 10 * (n_good + 1);
}

vector<pair<RockSampleState, double>> RockSample::InitialBelief() const {
  vector<pair<RockSampleState, double>> belief;
  for (int k = 0; k < (1 << K_); k++)
    belief.push_back({ MakeState(robot_start_cell_, k), 1.0 / (1 << K_) });
  return belief;
}

void RockSample::PrintState(const RockSampleState& state, ostream& out) const {
  int ac = CellOf(state);
  for (int i = 0; i < size_; i++) {
    for (int j = 0; j < size_; j++) {
      if (ac == CellNum(i, j)) {
        if (rock_at_cell_[ac] == -1) 
          out << "R ";
        else if (RockStatus(rock_at_cell_[ac], RockSetOf(state)))
          out << "G ";
        else
          out << "B ";
        continue;
      }
      if (rock_at_cell_[CellNum(i, j)] == -1) 
        out << ". ";
      else if (RockStatus(rock_at_cell_[CellNum(i, j)], RockSetOf(state)))
        out << "1 ";
      else
        out << "0 ";
    }
    out << endl;
  }
}

vector<vector<UMAP<RockSampleState, double>>> RockSample::TransitionMatrix() 
    const {
  vector<vector<UMAP<RockSampleState, double>>> ans(NumStates());
  for (int i = 0; i < NumStates(); i++) {
    ans[i].resize(NumActions());
    for (int j = 0; j < NumActions(); j++)
      ans[i][j] = { { T[i][j], 1 } }; 
  }
  return ans;
}

pair<double, int> RockSample::LowerBound(
    History& history,
    const vector<Particle<RockSampleState>*>& particles,
    int stream_position,
    const Model<RockSampleState>& model) const {
  // Strategy: Compute a representative state by setting the state of 
  // each rock to the one that occurs more frequently in the particle set.
  // Then compute the best sequence of actions for the resulting
  // state. Apply this sequence of actions to each particle and average
  // to get a lower bound. 
  //
  // Possible improvement: If a rock is sampled while replaying the action
  // sequence, use dynamic programming to look forward in the action
  // sequence to determine if it would be a better idea to first sense the
  // rock instead. (sensing eliminates the bad rocks in the particle set)

  if (IsTerminal(particles.front()->state))
    return {0, -1};

  bool debug = false;
  
  // The expected value of sampling a rock, over all particles
  vector<double> expected_sampling_value(K_);
  int seen_ptr = 0;

  // Compute the expected sampling value of each rock. Instead of factoring
  // the weight of each particle, we first record the weight of each state.
  // This is so that the inner loop that updates the expected value of each
  // rock runs once per state seen, instead of once per particle seen. If
  // there are lots of common states between particles, this gives a 
  // significant speedup to the search because the lower bound is the 
  // bottleneck.
  for (auto p: particles) {
    if (weight_sum_of_state_[p->state] == -Globals::INF) {
      weight_sum_of_state_[p->state] = p->wt;
      state_seen_[seen_ptr++] = p->state;
    }
    else
      weight_sum_of_state_[p->state] += p->wt;
  }
  double ws = 0;
  for (int i = 0; i < seen_ptr; i++) {
    RockSampleState s = state_seen_[i];
    ws += weight_sum_of_state_[s];
    for (int i = 0; i < K_; i++)
      expected_sampling_value[i] += 
        weight_sum_of_state_[s] * (RockStatus(i, s) ? 10 : -10);
    // Reset for next use
    weight_sum_of_state_[s] = -Globals::INF;
  }

  int most_likely_rock_set = 0;
  for (int i = 0; i < K_; i++) {
    expected_sampling_value[i] /= ws;
    // Threshold the average to good or bad
    if (expected_sampling_value[i] > -Globals::TINY)
      most_likely_rock_set |= (1 << i);
    if (Globals::Fequals(0, expected_sampling_value[i]))
      expected_sampling_value[i] = 0;
    if (debug)
      cerr << "ESV[" << i << "] = " << expected_sampling_value[i] << endl;
  }
  if (debug)
    cerr << "MLRockSample = " << most_likely_rock_set << endl;
  RockSampleState most_likely_state = MakeState(
      CellOf(particles.front()->state),
      most_likely_rock_set);
  RockSampleState s = most_likely_state;

  // Sequence of actions taken in the optimal policy
  vector<int> optimal_policy;
  pii prev_cell;
  double ret;
  double reward;
  while (1) {
    int act = (*upper_bound_act_)[s];
    if (debug) { cerr << act << " "; }
    RockSampleState s_ = s;
    Step(s_, 0/*dummy*/, act, reward);
    if (IsTerminal(s_)) {
      prev_cell = cell_to_coords_[CellOf(s)]; 
      ret = 10;
      break;
    }
    optimal_policy.push_back(act);
    if (optimal_policy.size() == Globals::config.search_depth) {
      prev_cell = cell_to_coords_[CellOf(s_)]; 
      ret = 0;
      break;
    }
    s = s_;
  }
  if (debug) cerr << endl;

  int best_a = optimal_policy.empty() ? 2 : optimal_policy[0];

  // Execute the sequence backwards to allow using the DP trick mentioned
  // earlier.
  for (int i = optimal_policy.size() - 1; i >= 0; i--) {
    int act = optimal_policy[i];
    ret *= Globals::config.discount;
    if (act == 4) {
      int rock = rock_at_cell_[CellNum(prev_cell.first, prev_cell.second)];
      // Uncomment the part below to enable sensing the rock before sampling
      // it. This is commented because it gives a rather good lower bound, 
      // making it difficult to assess the effectiveness of the search.
      /*
      double valueWithSensing = 
        Globals::config.discount * (10 + expectedSamplingValue[rock]) / 2;
      double maybe1 = valueWithSensing + Globals::config.discount * ret;
      double maybe2 = expectedSamplingValue[rock] + ret;
      if (maybe1 > maybe2) {
        ret = maybe1;
        if (i == 0)
          bestA = 5 + rock;
      }
      else
        ret = maybe2;
      */
      ret = expected_sampling_value[rock] + ret;
      continue;
    }
    // Move in the opposite direction since we're going backwards
    switch(act) {
      case 0: prev_cell.first++; break; 
      case 1: prev_cell.first--; break; 
      case 2: prev_cell.second--; break; 
      case 3: prev_cell.second++; break; 
      default: assert(false);
    }
  }

  return {ret, best_a};
}
