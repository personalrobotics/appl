#include "tag.h"

Tag::Tag(unsigned initial_state_seed) {
  int map[5][10] = {
    { 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 },
    { 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 },
    { 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
  };

  int cell_counter = 0;
  for (int i = 0; i < 5; i++) {
    floor_map_.push_back(vector<int>(10));
    for (int j = 0; j < 10; j++)
      floor_map_[i][j] = map[i][j] ? cell_counter++ : -1;
  }

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
  T_.resize(n_states_);
  for (int rob = 0; rob < n_cells_; rob++) {
    for (int opp = 0; opp < n_cells_; opp++) {
      TagState s = CellPairToState(rob, opp);
      T_[s].resize(NumActions());
      UMAP<int, double> o_next_cells = GetONextCells(rob, opp);
      // Motion actions
      for (int a = 0; a < TagAct(); a++) {
        int r_next_cell = GetRNextCell(rob, a);
        for (auto& it : o_next_cells) {
          TagState s_ = CellPairToState(r_next_cell, it.first);
          T_[s][a][s_] = it.second;
        }
      }
      // Tag action
      if (rob == opp)
        T_[s][TagAct()][CellPairToState(rob, n_cells_)] = 1.0;
      else {
        for (auto& it : o_next_cells) {
          TagState s_ = CellPairToState(rob, it.first);
          T_[s][TagAct()][s_] = it.second;
        }
      }
    }
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
}

int Tag::GetRNextCell(int r, int a) const {
  static int dx[] = {-1, 1, 0, 0}, dy[] = {0, 0, 1, -1};
  pii r_coord = cell_to_coords_[r];
  int tx = r_coord.first + dx[a];
  int ty = r_coord.second + dy[a];
  if (WithinMap(tx, ty) && floor_map_[tx][ty] != -1)
    return floor_map_[tx][ty];
  return r;
}

void Tag::Step(TagState& s, double random_num, int action, double& reward) 
    const {
  auto& probs = T_[s][action];
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

void Tag::Step(TagState& s, double random_num, int action, double& reward, 
               uint64_t& obs) const {
  auto& probs = T_[s][action];
  double sum = 0;
  for (auto& it : probs) {
    sum += it.second;
    if (sum > random_num) {
      s = it.first;
      break;
    }
  }
  if (SameLocation(s)) {
    if (IsTerminal(s)) {
      reward = TAG_REWARD;
      obs = TerminalObs();
    } 
    else {
      reward = action == TagAct() ? -TAG_REWARD : -1;
      obs = n_cells_;
    }
  }
  else {
    reward = action == TagAct() ? -TAG_REWARD : -1;
    obs = state_to_cell_pair_[s].first;
  }
}

double Tag::ObsProb(uint64_t obs, const TagState& s, int a) const {
  if (IsTerminal(s))
    return obs == TerminalObs();
  if (SameLocation(s))
    return obs == n_cells_;
  return obs == state_to_cell_pair_[s].first;
}

UMAP<int, double> Tag::GetONextCells(int rob_, int opp_) const {
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

double Tag::FringeUpperBound(const TagState& s) const {
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

int Tag::DefaultActionPreferred(const History& history, const TagState& state) 
    const {

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

vector<pair<TagState, double>> Tag::InitialBelief() const {
  vector<pair<TagState, double>> belief;
  for (int i = 0; i < n_cells_; i++)
    for (int j = 0; j < n_cells_; j++)
      belief.push_back({ CellPairToState(i, j), 1.0 / (n_cells_ * n_cells_) });
  return belief;
}

void Tag::PrintState(const TagState& state, ostream& out) const {
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
