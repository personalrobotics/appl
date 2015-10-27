#include "lasertag.h"

uint64_t LaserTag::same_loc_obs_;
uint64_t LaserTag::terminal_obs_;

LaserTag::LaserTag(unsigned initial_state_seed) {
  int nrows = 7;
  int ncols = 11;
  int nblock = 8;
  
  for (int i = 0; i < nrows; i++)
    floor_map_.push_back(vector<int>(ncols));

  do {
    int block_x = rand_r(&initial_state_seed) % nrows;
    int block_y = rand_r(&initial_state_seed) % ncols;
    if (floor_map_[block_x][block_y] != -1) {
      floor_map_[block_x][block_y] = -1;
      nblock--;
    }
  } while (nblock);

  int cell_counter = 0;
  for (int i = 0; i < nrows; i++)
    for (int j = 0; j < ncols; j++)
      if (floor_map_[i][j] != -1)
        floor_map_[i][j] = cell_counter++;

  n_cells_ = 0;
  for (int i = 0; i < floor_map_.size(); i++)
    for (int j = 0; j < floor_map_[0].size(); j++)
      n_cells_ = max(n_cells_, floor_map_[i][j]);
  n_cells_++;

  rob_start_cell_ = rand_r(&initial_state_seed) % n_cells_;
  opp_start_cell_ = rand_r(&initial_state_seed) % n_cells_;

  n_states_ = n_cells_ * (n_cells_ + 1);

  cell_to_coords_.resize(n_cells_);
  for (int i = 0; i < floor_map_.size(); i++)
    for (int j = 0; j < floor_map_[0].size(); j++)
      if (floor_map_[i][j] != -1)
        cell_to_coords_[floor_map_[i][j]] = {i, j};

  state_to_coord_pair_.resize(n_states_);
  state_to_cell_pair_.resize(n_states_);

  /* Build map from state to pair of coordinates */
  // Non-terminal states
  for (int i = 0; i < n_cells_; i++) {
    for (int j = 0; j < n_cells_; j++) {
      pair<pii, pii> p = {cell_to_coords_[i], cell_to_coords_[j]};
      state_to_cell_pair_[CellPairToState(i, j)] = {i, j};
      state_to_coord_pair_[CellPairToState(i, j)] = p;
    }
  }
  // Terminal states
  for (int i = 0; i < n_cells_; i++) {
    pii coords = cell_to_coords_[i];
    state_to_cell_pair_[CellPairToState(i, n_cells_)] = {i, n_cells_};
    state_to_coord_pair_[CellPairToState(i, n_cells_)] = {coords, coords};
  }

  // Initialize the transition matrix T 
  T.resize(n_states_);
  for (int rob = 0; rob < n_cells_; rob++) {
    for (int opp = 0; opp < n_cells_; opp++) {
      LaserTagState s = CellPairToState(rob, opp);
      T[s].resize(NumActions());
      UMAP<int, double> o_next_cells = GetONextCells(rob, opp);
      // Motion actions
      for (int a = 0; a < TagAct(); a++) {
        int r_next_cell = GetRNextCell(rob, a);
        for (auto& it : o_next_cells) {
          LaserTagState s_ = CellPairToState(r_next_cell, it.first);
          T[s][a][s_] = it.second;
        }
      }
      // Tag action
      if (rob == opp)
        T[s][TagAct()][CellPairToState(rob, n_cells_)] = 1.0;
      else {
        for (auto& it : o_next_cells) {
          LaserTagState s_ = CellPairToState(rob, it.first);
          T[s][TagAct()][s_] = it.second;
        }
      }
    }
  }

  // Initialize true laser distances for each state
  vector<pii> dirs = { {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
                       {1, 0}, {1, -1}, {0, -1}, {-1, -1} };
  laser_dist_.resize(n_states_);
  for (int s = 0; s < n_states_; s++) {
    laser_dist_[s].resize(NBEAMS);
    if (SameLocation(s)) {
      fill(laser_dist_[s].begin(), laser_dist_[s].end(), 0);
      continue;
    }
    int rob = state_to_cell_pair_[s].first;
    int opp = state_to_cell_pair_[s].second;
    // Block opponents cell
    floor_map_[cell_to_coords_[opp].first][cell_to_coords_[opp].second] = -1;
    for (int d = 0; d < dirs.size(); d++) {
      int tx = cell_to_coords_[rob].first;
      int ty = cell_to_coords_[rob].second;
      double curdist = 0.0;
      // Take one extra to avoid 0 distances
      while (WithinMap(tx, ty)) {
        tx += dirs[d].first;
        ty += dirs[d].second;
        curdist += sqrt(pow(dirs[d].first, 2) + pow(dirs[d].second, 2));
      }
      laser_dist_[s][d] = curdist;
    }
    floor_map_[cell_to_coords_[opp].first][cell_to_coords_[opp].second] = opp;
  }

  best_default_action_memo_.resize(n_states_);
  for (int s = 0; s < n_states_; s++) {
    if (SameLocation(s)) {
      best_default_action_memo_[s] = TagAct();
      continue;
    }
    pii robot = state_to_coord_pair_[s].first; 
    pii target = state_to_coord_pair_[s].second; 
    vector<int> a_star; // Order actions according to goodness
    double xdiff = target.first - robot.first;
    double ydiff = target.second - robot.second;
    a_star.push_back(xdiff < 0 ? 0 : 1);
    a_star.push_back(ydiff > 0 ? 2 : 3);
    if (abs(ydiff) > abs(xdiff))
      swap(a_star[0], a_star[1]);
    for (int a = 0; a < TagAct(); a++) {
      if (find(a_star.begin(), a_star.end(), a) == a_star.end())
        a_star.push_back(a);
    }
    // Select the best one *possible*, or 0 by default
    int best_a = 0;
    for (int i = 0; i < TagAct(); i++) {
      int next_cell = GetRNextCell(floor_map_[robot.first][robot.second],
                                   a_star[i]);
      if (next_cell != floor_map_[robot.first][robot.second]) {
        best_a = a_star[i];
        break;
      }
    }
    best_default_action_memo_[s] = best_a;
  }

  reading_distributions_.resize(n_states_);
  for (int s = 0; s < n_states_; s++)
    InitReadingDistribution(s);

  // Compute terminalObs_
  for (int i = 0; i < NBEAMS; i++)
    SetReading(terminal_obs_, 100, i);
  for (int i = 0; i < NBEAMS; i++)
    SetReading(same_loc_obs_, 101, i);
}

void LaserTag::InitReadingDistribution(int s) {
  vector<double>& actual = laser_dist_[s];
  reading_distributions_[s].resize(NBEAMS); 

  for (int d = 0; d < NBEAMS; d++) {
    for (int reading = 0; reading < actual[d]/unit_size_; reading++) {
      double left = reading * unit_size_;
      double right = min(actual[d], (reading + 1) * unit_size_);
      left -= actual[d];
      right -= actual[d];
      double prob_mass = 2 * (Cdf(right, 0, noise_sigma_) -
                             (reading ? Cdf(left, 0, noise_sigma_) : 0));

      reading_distributions_[s][d].push_back(prob_mass);
    }
  }
}

/* Magic. Do not touch. */
double LaserTag::Erf(double x) {
  // constants
  double a1 =  0.254829592;
  double a2 = -0.284496736;
  double a3 =  1.421413741;
  double a4 = -1.453152027;
  double a5 =  1.061405429;
  double p  =  0.3275911;
  // Save the sign of x
  int sign = 1;
  if (x < 0)
    sign = -1;
  x = fabs(x);
  // A&S formula 7.1.26
  double t = 1.0 / (1.0 + p*x);
  double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

  return sign * y;
}

// CDF of the normal distribution
double LaserTag::Cdf(double x, double mean, double sigma) {
  return 0.5 * (1 + erf((x - mean) / (sqrt(2) * sigma)));
}

int LaserTag::DefaultActionPreferred(const History& history, 
    const LaserTagState& state) const {

  static vector<int> actions(NumActions());

  // If history is empty then we don't have any preference
  if (history.Size() == 0)
    return -1;

  // If we just saw an opponent then tag
  if (history.LastObservation() == n_cells_)
    return 4;
  
  // Don't double back and don't go into walls
  int num_actions = 0;
  for (int d = 0; d < 4; d++) {
    pii agent_pos = 
        cell_to_coords_[GetRNextCell(state_to_cell_pair_[state].first, d)];
    if (!Opposite(history.LastAction(), d) && 
        WithinMap(agent_pos.first, agent_pos.second))
      actions[num_actions++] = d;
  }

  if (num_actions) {
    srand(history.Hash());
    return actions[rand() % num_actions];
  }

  return -1;
}

int LaserTag::GetRNextCell(int r, int a) const {
  static int dx[] = {-1, 1, 0, 0}, dy[] = {0, 0, 1, -1};
  pii r_coord = cell_to_coords_[r];
  int tx = r_coord.first + dx[a];
  int ty = r_coord.second + dy[a];
  if (WithinMap(tx, ty) && floor_map_[tx][ty] != -1)
    return floor_map_[tx][ty];
  return r;
}

void LaserTag::Step(LaserTagState& s, double random_num, int action, 
                                double& reward) const {
  auto& probs = T[s][action];
  double sum = 0;
  for (auto& it : probs) {
    sum += it.second;
    if (sum > random_num) {
      s = it.first;
      break;
    }
  }
  if (IsTerminal(s))
    reward = TAG_REWARD;
  else
    reward = action == TagAct() ? -TAG_REWARD : -1;
}

void LaserTag::Step(LaserTagState& s, double random_num, int action, 
                                double& reward, uint64_t& obs) const {
  auto& probs = T[s][action];
  double sum = 0;
  for (auto& it : probs) {
    sum += it.second;
    if (sum > random_num) {
      s = it.first;
      break;
    }
  }
  if (IsTerminal(s)) {
    reward = TAG_REWARD;
    obs = TerminalObs();
  } 
  else {
    reward = action == TagAct() ? -TAG_REWARD : -1;
    if (SameLocation(s)) {
      obs = same_loc_obs_;
    }
    else {
      obs = 0;
      // Get noisy laser readings
      unsigned int rng_seed = ceil(random_num * RAND_MAX);
      for (int dir = 0; dir < NBEAMS; dir++) {
        double mass = 1.0 * rand_r(&rng_seed) / RAND_MAX;
        int reading = 0;
        for (; reading < reading_distributions_[s][dir].size(); reading++) {
          mass -= reading_distributions_[s][dir][reading];
          if (mass < Globals::TINY)
            break;
        }
        SetReading(obs, reading, dir);
      }
    }
  }
}

double LaserTag::ObsProb(uint64_t o, const LaserTagState& s, int a)
    const {
  // if either is true, both have to be true so return 0 or 1
  if (IsTerminal(s) || (o == TerminalObs()))
    return (o == TerminalObs() && IsTerminal(s));

  // if either is true, both have to be true so return 0 or 1
  if (SameLocation(s) || (o == same_loc_obs_))
    return (o == same_loc_obs_) && SameLocation(s);

  const vector<double>& actual = laser_dist_[s];
  double prod = 1.0;
  for (int d = 0; d < NBEAMS; d++) {
    int reading = GetReading(o, d);
    if (reading * unit_size_ >= actual[d])
      return 0;
    double prob_mass = reading_distributions_[s][d][reading];
    prod *= prob_mass;
  }

  return prod;
}

UMAP<int, double> LaserTag::GetONextCells(int rob_, int opp_) const {
  pii rob = cell_to_coords_[rob_];
  pii opp = cell_to_coords_[opp_];
  UMAP<int, double> ret;

  // Possible row movements
  if (opp.first != rob.first) {
    int x_ = opp.first + (opp.first > rob.first ? 1 : -1);
    if (!WithinMap(x_, opp.second))
      x_ = opp.first;
    ret[floor_map_[x_][opp.second]] += 0.4;
  }
  else {
    int x_ = WithinMap(opp.first-1, opp.second) ? opp.first-1 : opp.first;
    ret[floor_map_[x_][opp.second]] += 0.2;
    x_ = WithinMap(opp.first+1, opp.second) ? opp.first+1 : opp.first;
    ret[floor_map_[x_][opp.second]] += 0.2;
  }

  // Possible column movements
  if (opp.second != rob.second) {
    int y_ = opp.second + (opp.second > rob.second ? 1 : -1);
    if (!WithinMap(opp.first, y_))
      y_ = opp.second;
    ret[floor_map_[opp.first][y_]] += 0.4;
  }
  else {
    int y_ = WithinMap(opp.first, opp.second-1) ? opp.second-1 : opp.second;
    ret[floor_map_[opp.first][y_]] += 0.2;
    y_ = WithinMap(opp.first, opp.second+1) ? opp.second+1 : opp.second;
    ret[floor_map_[opp.first][y_]] += 0.2;
  }
  ret[opp_] += 0.2;

  return ret;
}

double LaserTag::FringeUpperBound(const LaserTagState& s) const {
  if (IsTerminal(s))
    return 0;

  // Manhattan distance
  auto cp = state_to_coord_pair_[s];
  int d = abs(cp.first.first - cp.second.first) +
          abs(cp.first.second - cp.second.second);
  if (Globals::config.discount < 1)
    return - (1 - pow(Globals::config.discount, d)) / (1 - Globals::config.discount) +
           TAG_REWARD * pow(Globals::config.discount, d);
  else
    return -d + TAG_REWARD;
}

vector<pair<LaserTagState, double>> LaserTag::InitialBelief() 
    const {
  vector<pair<LaserTagState, double>> belief;
  for (int i = 0; i < n_cells_; i++)
    belief.push_back({ CellPairToState(rob_start_cell_, i), 1.0 / n_cells_ });
  return belief;
}

void LaserTag::PrintState(const LaserTagState& state, 
                                      ostream& out) const {
  pii cell_pair = state_to_cell_pair_[state];
  for (int i = 0; i < floor_map_.size(); i++) {
    for (int j = 0; j < floor_map_[0].size(); j++) {
      if (floor_map_[i][j] == -1) 
        out << "#";
      else if (floor_map_[i][j] == cell_pair.first)
        out << (IsTerminal(CellPairToState(cell_pair.first, cell_pair.second))
                ? "$"
                : "R");
      else if (floor_map_[i][j] == cell_pair.second)
        out << "O";
      else
        out << ".";
      out << " ";
    }
    out << endl;
  }
}
