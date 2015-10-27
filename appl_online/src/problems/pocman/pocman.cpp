#include "pocman.h"

Pocman::Pocman(unsigned initial_state_seed, const RandomStreams& streams)
    : IUpperBound<PocmanState>(streams),
      initial_state_seed_(initial_state_seed),
      passage_y_(-1),
      smell_range_(1),
      hear_range_(2),
      food_prob_(0.5),
      chase_prob_(0.75),
      defensive_slip_(0.25),
      reward_clear_level_(+1000),
      reward_default_(-1),
      reward_die_(-100),
      reward_eat_food_(+10),
      reward_eat_ghost_(+25),
      reward_hit_wall_(-25),
      power_num_steps_(15) {

  InitMazeFull();
  InitLookupTables();
}

void Pocman::InitMazeMicro() {
  int maze[7][7] = {
    { 3, 3, 3, 3, 3, 3, 3 },
    { 3, 3, 0, 3, 0, 3, 3 },
    { 3, 0, 3, 3, 3, 0, 3 },
    { 3, 3, 3, 0, 3, 3, 3 },
    { 3, 0, 3, 3, 3, 0, 3 },
    { 3, 3, 0, 3, 0, 3, 3 },
    { 3, 3, 3, 3, 3, 3, 3 }
  };

  maze_ = Grid<int>(7, 7);
  for (int x = 0; x < 7; x++)
    maze_.SetCol(x, maze[x]);
  num_ghosts_ = 1;
  ghost_range_ = 3;
  pocman_home_ = Coord(3, 0);
  ghost_home_ = Coord(3, 4);
}

void Pocman::InitMazeMini() {
  int maze[10][10] = {
    { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 3, 0, 0, 3, 0, 0, 3, 0, 0, 3 },
    { 3, 0, 3, 3, 3, 3, 3, 3, 0, 3 },
    { 3, 3, 3, 0, 0, 0, 0, 3, 3, 3 },
    { 0, 0, 3, 0, 1, 1, 3, 3, 0, 0 },
    { 0, 0, 3, 0, 1, 1, 3, 3, 0, 0 },
    { 3, 3, 3, 0, 0, 0, 0, 3, 3, 3 },
    { 3, 0, 3, 3, 3, 3, 3, 3, 0, 3 },
    { 3, 0, 0, 3, 0, 0, 3, 0, 0, 3 },
    { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 }
  };

  maze_ = Grid<int>(10, 10);
  for (int x = 0; x < 10; x++)
    maze_.SetCol(x, maze[x]);
  num_ghosts_ = 3;
  ghost_range_ = 4;
  pocman_home_ = Coord(4, 2);
  ghost_home_ = Coord(4, 4);
  passage_y_ = 5;
}

void Pocman::InitMazeFull() {
  // Transposed maze
  int maze[19][17] = {
    { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, },
    { 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3, },
    { 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, },
    { 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, },
    { 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, },
    { 0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, },
    { 0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0, },
    { 0, 0, 0, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 0, 0, 0, },
    { 1, 1, 1, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 1, 1, 1, },
    { 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, },
    { 0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0, },
    { 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, },
    { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, },
    { 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3, },
    { 7, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 7, },
    { 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, },
    { 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, },
    { 3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, },
    { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3  }
  };

  maze_ = Grid<int>(17, 19);
  // Transpose to rows
  for (int x = 0; x < 19; x++)
    maze_.SetRow(x, maze[18 - x]);
  num_ghosts_ = 4;
  ghost_range_ = 6;
  pocman_home_ = Coord(8, 6);
  ghost_home_ = Coord(8, 10);
  passage_y_ = 10;
}

PocmanState Pocman::GetStartState() const {
  return GetStartState(initial_state_seed_);
}

PocmanState Pocman::GetStartState(unsigned seed) const {
  PocmanState state;
  state.ghost_pos_.resize(num_ghosts_);
  state.ghost_dir_.resize(num_ghosts_);
  state.food_.resize(maze_.GetXSize() * maze_.GetYSize());
  NewLevel(state, seed);
  return state;
}

vector<pair<PocmanState, double>> Pocman::InitialBelief() const {
  // As the number of states is very large, we return some random states
  unsigned seed = initial_state_seed_ + 1;
  vector<pair<PocmanState, double>> belief;
  for (int i = 0; i < Globals::config.n_particles; i++)
    belief.push_back({ GetStartState(rand_r(&seed)), 
                       1.0 / Globals::config.n_particles });
  return belief;
}

void Pocman::NewLevel(PocmanState& state, unsigned& seed) const {
  state.pocman_pos_ = pocman_home_;
  for (int g = 0; g < num_ghosts_; g++) {
    state.ghost_pos_[g] = ghost_home_;
    state.ghost_pos_[g].X += g % 2;
    state.ghost_pos_[g].Y += g / 2;
    state.ghost_dir_[g] = -1;
  }

  state.num_food_ = 0;
  for (int x = 0; x < maze_.GetXSize(); x++) {
    for (int y = 0; y < maze_.GetYSize(); y++) {
      int poc_index = maze_.Index(x, y);
      if (CheckFlag(maze_(x, y), E_SEED) &&
          (CheckFlag(maze_(x, y), E_POWER) ||
           rand_r(&seed) < food_prob_ * RAND_MAX)) {
        state.food_[poc_index] = 1;
        state.num_food_++;
      }
      else
        state.food_[poc_index] = 0;
    }
  }

  state.power_steps_ = 0;
}

void Pocman::InitLookupTables() {
  int sx = maze_.GetXSize(), sy = maze_.GetYSize();

  // nextpos_table_: Next cell of pocman given a starting point and a direction
  nextpos_table_.resize(sx);
  for (int i = 0; i < sx; i++) {
    nextpos_table_[i].resize(sy);
    for (int j = 0; j < sy; j++) {
      nextpos_table_[i][j].resize(8);

      Coord from(i, j), nextpos;
      for (int dir = 0; dir < 8; dir++) {
        if (from.X == 0 && from.Y == passage_y_ && dir == Coord::E_WEST)
          nextpos = Coord(sx - 1, from.Y);
        else if (from.X == maze_.GetXSize() - 1 && from.Y == passage_y_
                && dir == Coord::E_EAST)
          nextpos = Coord(0, from.Y);
        else
          nextpos = from + Coord::Compass[dir];

        if (maze_.Inside(nextpos) && Passable(nextpos))
          nextpos_table_[from.X][from.Y][dir] = nextpos;
        else
          nextpos_table_[from.X][from.Y][dir] = Coord::Null;
      }
    }
  }

  // relative_dir_table_: which direction a cell is from pocman (-1 if not NESW)
  relative_dir_table_.resize(sx);
  for (int i = 0; i < sx; i++) {
    relative_dir_table_[i].resize(sy);
    for (int j = 0; j < sy; j++) {

      Coord pocman_pos(i, j);
      if (!Passable(pocman_pos))
        continue;

      relative_dir_table_[i][j].resize(sx);
      for (int k = 0; k < sx; k++) {
        relative_dir_table_[i][j][k].resize(sy);
        for (int l = 0; l < sy; l++)
          relative_dir_table_[i][j][k][l] = -1;
      }

      for (int dir = 0; dir < 4; dir++) {
        Coord eyepos = pocman_pos + Coord::Compass[dir];
        while (maze_.Inside(eyepos) && Passable(eyepos)) {
          relative_dir_table_[i][j][eyepos.X][eyepos.Y] = dir;
          eyepos += Coord::Compass[dir];
        }
      }
    }
  }

  // smellfood_table_: Which cells need to be checked for presence of food,
  // given pocman's position
  smellfood_table_.resize(sx);
  for (int i = 0; i < sx; i++) {
    smellfood_table_[i].resize(sy);
    for (int j = 0; j < sy; j++) {
      Coord pocman_pos(i, j);
      if (!Passable(pocman_pos))
        continue;
      for (int x = -smell_range_; x <= smell_range_; x++)
        for (int y = -smell_range_; y <= smell_range_; y++)  {
          Coord smell_pos = pocman_pos + Coord(x, y);
          if (maze_.Inside(smell_pos) && Passable(smell_pos))
            smellfood_table_[i][j].push_back(maze_.Index(smell_pos));
        }
    }
  }

  // moveghost_random_table_: List of possible next positions for a ghost,
  // given an initial position and direction (direction -1 indexed as 4)
  moveghost_random_table_.resize(sx);
  for (int i = 0; i < sx; i++) {
    moveghost_random_table_[i].resize(sy);
    for (int j = 0; j < sy; j++) {
      Coord pos(i, j);
      if (!Passable(pos)) continue;
      moveghost_random_table_[i][j].resize(5);
      for (int dir = -1; dir < 4; dir++) {
        // Never switch to opposite direction
        // Currently assumes there are no dead-ends.
        for (int newdir = 0; newdir < 4; newdir++) {
          if (Coord::Opposite(dir) == newdir) continue;
          Coord newpos = nextpos_table_[pos.X][pos.Y][newdir];
          if (newpos.Valid())
            moveghost_random_table_[i][j][dir == -1 ? 4 : dir].push_back(
                {newdir, newpos});
        }
      }
    }
  }

  // preferred_powerpill_table_: set of preferred actions when under the
  // effect of a powerpill
  preferred_powerpill_table_.resize(1 << 4);
  for (int obs = 0; obs < (1 << 4); obs++)
    for (int a = 0; a < 4; a++)
      if (CheckFlag(obs, a))
        preferred_powerpill_table_[obs].push_back(a);

  // preferred_normal_table_: set of preferred actions otherwise (avoid
  // running into ghosts and changing directions). indexed by position of
  // pocman, last observation, and last action
  preferred_normal_table_.resize(sx * sy);
  for (int i = 0; i < sx; i++)
    for (int j = 0; j < sy; j++) {
      Coord pos(i, j);
      if (!Passable(pos)) continue;
      int idx = maze_.Index(pos);
      preferred_normal_table_[idx].resize(1 << 4);
      for (int obs = 0; obs < (1 << 4); obs++) {
        preferred_normal_table_[idx][obs].resize(4);
        for (int prev_action = 0; prev_action < 4; prev_action++)
          for (int a = 0; a < 4; a++) {
            if (Coord::Opposite(a) == prev_action ||
               CheckFlag(obs, a)) continue;
            Coord newpos = nextpos_table_[pos.X][pos.Y][a];
            if (newpos.Valid())
              preferred_normal_table_[idx][obs][prev_action].push_back(a);
          }
      }
    }

  // wall_obs_table_: Whether pocman can feel a wall in 4 directions for any
  // given position
  wall_obs_table_.resize(sx);
  for (int i = 0; i < sx; i++) {
    wall_obs_table_[i].resize(sy);
    for (int j = 0; j < sy; j++) {
      Coord pos(i, j);
      if (!Passable(pos))
        continue;
      int bits = 0;
      for (int d = 0; d < 4; d++) {
        Coord wpos = nextpos_table_[pos.X][pos.Y][d];
        if (wpos.Valid() && Passable(wpos))
          SetFlag(bits, d + 4);
      }
      wall_obs_table_[i][j] = bits;
    }
  }
}

inline
uint64_t Pocman::MakeObservation(const PocmanState& state) const {
  if (state.terminal_)
    return TerminalObs();

  int observation = 0;

  for (int d = 0; d < 4; d++)
    if (SeeGhost(state, d) >= 0)
      SetFlag(observation, d);

  SetWallObs(observation, state);

  if (SmellFood(state))
    SetFlag(observation, 8);

  if (HearGhost(state))
    SetFlag(observation, 9);

  return observation;
}

void Pocman::Step(PocmanState& state, double rand_num, int action,
                  double& reward, uint64_t& observation) const {
  reward = reward_default_;

  Coord newpos = NextPos(state.pocman_pos_, action);
  if (newpos.Valid())
    state.pocman_pos_ = newpos;
  else
    reward += reward_hit_wall_;

  if (state.power_steps_ > 0)
    --state.power_steps_;

  unsigned rand_seed = rand_num * RAND_MAX;
  for (int g = 0; g < num_ghosts_; g++) {
    bool hit_ghost = false;
    if (state.ghost_pos_[g] == state.pocman_pos_)
      hit_ghost = true;
    MoveGhost(state, g, (double)(rand_r(&rand_seed)) / RAND_MAX);
    if (state.ghost_pos_[g] == state.pocman_pos_)
      hit_ghost = true;
    if (hit_ghost) {
      if (state.power_steps_ > 0) {
        reward += reward_eat_ghost_;
        state.ghost_pos_[g] = ghost_home_;
        state.ghost_dir_[g] = -1;
      }
      else {
        reward += reward_die_;
        state.terminal_ = true;
        observation = TerminalObs();
        return;
      }
    }
  }

  int poc_index = maze_.Index(state.pocman_pos_);
  if (state.food_[poc_index] == 1) {
    EatFoodInCurrentCell(state);
    reward += reward_eat_food_;
    if (state.num_food_ == 0) {
      reward += reward_clear_level_;
      state.terminal_ = true;
      observation = TerminalObs();
      return;
    }
  }

  observation = MakeObservation(state);
}

double Pocman::ObsProb(uint64_t z, const PocmanState& state, int action)
    const {
  return z == MakeObservation(state);
}

inline
void Pocman::MoveGhost(PocmanState& state, int g, double rand_num) const {
  if (Coord::ManhattanDistance(
        state.pocman_pos_, state.ghost_pos_[g]) < ghost_range_) {
    if (state.power_steps_ > 0)
      MoveGhostDefensive(state, g, rand_num);
    else
      MoveGhostAggressive(state, g, rand_num);
  }
  else {
    MoveGhostRandom(state, g, rand_num);
  }
}

void Pocman::MoveGhostAggressive(PocmanState& state, int g, double rand_num)
    const {
  if (rand_num > chase_prob_) {
    MoveGhostRandom(state, g, rand_num);
    return;
  }

  int best_dist = maze_.GetXSize() + maze_.GetYSize();
  Coord best_pos = state.ghost_pos_[g];
  int best_dir = -1;
  for (int dir = 0; dir < 4; dir++) {
    int dist = Coord::DirectionalDistance(
        state.pocman_pos_, state.ghost_pos_[g], dir);
    Coord newpos = NextPos(state.ghost_pos_[g], dir);
    if (dist <= best_dist && newpos.Valid()
        && Coord::Opposite(dir) != state.ghost_dir_[g]) {
      best_dist = dist;
      best_pos = newpos;
    }
  }

  state.ghost_pos_[g] = best_pos;
  state.ghost_dir_[g] = best_dir;
}

void Pocman::MoveGhostDefensive(PocmanState& state, int g, double rand_num)
    const {
  if (rand_num < defensive_slip_ && state.ghost_dir_[g] >= 0) {
    state.ghost_dir_[g] = -1;
    return;
  }

  int best_dist = 0;
  Coord best_pos = state.ghost_pos_[g];
  int best_dir = -1;
  for (int dir = 0; dir < 4; dir++) {
    int dist = Coord::DirectionalDistance(
        state.pocman_pos_, state.ghost_pos_[g], dir);
    Coord newpos = NextPos(state.ghost_pos_[g], dir);
    if (dist >= best_dist && newpos.Valid()
        && Coord::Opposite(dir) != state.ghost_dir_[g]) {
      best_dist = dist;
      best_pos = newpos;
    }
  }

  state.ghost_pos_[g] = best_pos;
  state.ghost_dir_[g] = best_dir;
}

inline
void Pocman::MoveGhostRandom(PocmanState& state, int g, double rand_num) const {
  int dir = state.ghost_dir_[g] == -1 ? 4 : state.ghost_dir_[g];
  auto& next_positions = moveghost_random_table_[state.ghost_pos_[g].X]
                                                [state.ghost_pos_[g].Y]
                                                [dir];
  assert(next_positions.size());
  int idx = (int)(rand_num * RAND_MAX) % next_positions.size();
  state.ghost_dir_[g] = next_positions[idx].first;
  state.ghost_pos_[g] = next_positions[idx].second;
}

int Pocman::SeeGhost(const PocmanState& state, int dir) const {
  for (int g = 0; g < num_ghosts_; g++)
    if (dir == relative_dir_table_[state.pocman_pos_.X]
                                  [state.pocman_pos_.Y]
                                  [state.ghost_pos_[g].X]
                                  [state.ghost_pos_[g].Y])
      return g;
  return -1;
}

bool Pocman::HearGhost(const PocmanState& state) const {
  for (int g = 0; g < num_ghosts_; g++)
    if (Coord::ManhattanDistance(
        state.ghost_pos_[g], state.pocman_pos_) <= hear_range_)
      return true;
  return false;
}

bool Pocman::SmellFood(const PocmanState& state) const {
  for (auto idx: smellfood_table_[state.pocman_pos_.X][state.pocman_pos_.Y])
    if (state.food_[idx] == 1)
      return true;
  return false;
}

void Pocman::PrintState(const PocmanState& state, ostream& ostr) const {
  for (int x = 0; x < maze_.GetXSize() + 2; x++)
    ostr << "X ";
  ostr << endl;
  for (int y = maze_.GetYSize() - 1; y >= 0; y--) {
    if (y == passage_y_)
      ostr << "< ";
    else
      ostr << "X ";
    for (int x = 0; x < maze_.GetXSize(); x++) {
      Coord pos(x, y);
      int index = maze_.Index(pos);
      char c = ' ';
      if (!Passable(pos))
        c = 'X';
      if (state.food_[index] == 1)
        c = CheckFlag(maze_(x, y), E_POWER) ? '+' : '.';
      for (int g = 0; g < num_ghosts_; g++)
        if (pos == state.ghost_pos_[g])
          c = (pos == state.pocman_pos_ ? '@' :
              (state.power_steps_ == 0 ? 'A' + g : 'a' + g));
      if (pos == state.pocman_pos_)
        c = state.power_steps_ > 0 ? '!' : '*';
      ostr << c << ' ';
    }
    if (y == passage_y_)
      ostr << ">" << endl;
    else
      ostr << "X" << endl;
  }
  for (int x = 0; x < maze_.GetXSize() + 2; x++)
    ostr << "X ";
  ostr << endl;
}

int Pocman::DefaultActionPreferred(const History& history,
    const PocmanState& state) const {
  if (history.Size() == 0)
    return -1;

  int action = history.LastAction();
  int observation = history.LastObservation();
  unsigned seed = history.Hash();

  // If power pill and can see a ghost then chase it
  if (state.power_steps_ > 0 && ((observation & 15) != 0)) {
    auto& ref = preferred_powerpill_table_[observation & 15];
    return ref.empty() ? -1 : ref[rand_r(&seed) % ref.size()];
  }

  // Otherwise avoid observed ghosts and avoid changing directions
  else {
    auto& ref = preferred_normal_table_[maze_.Index(state.pocman_pos_)]
                                       [observation & 15]
                                       [action];
    return ref.empty() ? -1 : ref[rand_r(&seed) % ref.size()];
  }
}

double Pocman::UpperBound(History& history,
                          const vector<Particle<PocmanState>*>& particles,
                          int stream_position, const Model<PocmanState>& model)
    const {
  if(IsTerminal(particles[0]->state)) {
    for(auto p: particles)
      assert(IsTerminal(p->state));
    return 0;
  }

  bool double_back = false;
  if (history.Size() >= 2) {
    int a1 = history.Action(history.Size() - 1);
    int a2 = history.Action(history.Size() - 2);
    if (Coord::Opposite(a1) == a2)
      double_back = true;
  }

  double ans = 0;

  int power_steps_check = -1;

  for (auto p: particles) {
    auto& state = p->state;
    int max_dist = 0;

    // Sanity check: make sure all particles have the same value of power_steps
    if(power_steps_check == -1)
      power_steps_check = state.power_steps_;
    else
      assert(power_steps_check == state.power_steps_);

    for (int i = 0; i < state.food_.size(); i++) {
      if (state.food_[i] != 1)
        continue;
      Coord food_pos = maze_.IndexToCoord(i);
      int dist = Coord::ManhattanDistance(state.pocman_pos_, food_pos);
      ans += reward_eat_food_ * pow(Globals::config.discount, dist);
      max_dist = max(max_dist, dist);
    }

    // Clear level
    ans += reward_clear_level_ * pow(Globals::config.discount, max_dist);

    // Default move-reward
    ans += reward_default_ * (Globals::config.discount < 1
             ? (1 - pow(Globals::config.discount, max_dist)) /
               (1 - Globals::config.discount)
             : max_dist);

    // If pocman is chasing a ghost, encourage it
    if (state.power_steps_ > 0 && history.Size() &&
        (history.LastObservation() & 15) != 0) {
      int act = history.LastAction();
      int obs = history.LastObservation();
      if (CheckFlag(obs, act)) {
        bool seen_ghost = false;
        for (int dist = 1; !seen_ghost; dist++) {
          Coord ghost_pos = state.pocman_pos_ + Coord::Compass[act] * dist;
          for (int g = 0; g < num_ghosts_; g++)
            if (state.ghost_pos_[g] == ghost_pos) {
              ans += reward_eat_ghost_ * pow(Globals::config.discount, dist);
              seen_ghost = true;
              break;
            }
        }
      }
    }

    // Ghost penalties
    double dist = 0;
    for (int g = 0; g < num_ghosts_; g++)
      dist += Coord::ManhattanDistance(state.pocman_pos_,
                                       state.ghost_pos_[g]);
    ans += reward_die_ * pow(Globals::config.discount, dist / num_ghosts_);

    // Penalize for doubling back, but not so much as to prefer hitting a wall
    if (double_back)
      ans += reward_hit_wall_ / 2;
  }

  ans /= particles.size();

  return ans;
}

void Pocman::PrintObs(uint64_t obs, ostream& out) const {
  if (obs == TerminalObs())
    out << "TERMINAL";
  else
    for (int i = 0; i < 10; i++, obs >>= 1)
      out << (obs & 1);
}

void Pocman::EatFoodInCurrentCell(PocmanState& state) const {
  int poc_index = maze_.Index(state.pocman_pos_);
  state.food_[poc_index] = -1;
  state.num_food_--;
  if (CheckFlag(maze_(poc_index), E_POWER))
    state.power_steps_ = power_num_steps_;
}

PocmanState Pocman::MakeConsistent(const PocmanState& inc_state, uint64_t obs,
    unsigned& seed) const {
  // We modify the inconsistent state to make it consistent with the
  // observation. The following possibilities can arise:
  //
  // (a) The observation is terminal but not the state:
  // In this case, we just return a terminal state.
  //
  // (b) The see-ghost/hear-ghost observations mismatch:
  // We resolve this by moving around ghosts randomly but conservatively (trying
  // not to disturb them if not necessary). Let G be the set of all ghosts. 
  // First, keep the closest ghost in every direction that pocman can see a 
  // ghost (set S), and remove all ghosts in directions where it can't. Then, in
  // directions where pocman can see a ghost but one isn't present, place a 
  // ghost randomly from the set G - S (this set is prioritized; to be 
  // conservative, we first try to place ghosts that were in inconsistent 
  // positions w.r.t seeing or hearing, then if we run out of them, we pick 
  // ghosts from elsewhere in the map that had nothing to do with the 
  // inconsistency).
  //
  // With the directions all satisfied, the hear-ghost observation may still
  // be inconsistent. If pocman can hear a ghost but one isn't present in
  // range, put one randomly in range. If it's the other way round, move it out
  // (in both cases, take care to preserve directional observations that we
  // satisfied earlier).
  //
  // (c) The smell-food observations mismatch:
  // This can be resolved by iterating over all cells in smelling range:
  //
  // - If can't smell food, remove all food
  // - If can smell food, place food in each cell with probability food_prob_,
  // making as many complete passes over the cells as required until at least 
  // one food is placed. Make sure not to place food in cells pocman has visited
  // before (this can be inferred from the starting state and history, so 
  // doesn't use any unobservable data).

  PocmanState state = inc_state;

  if (obs == TerminalObs()) {
    assert(!state.terminal_);
    // Just return a state for which IsTerminal() is true
    state.terminal_ = true;
    return state;
  }

  else {
    state.terminal_ = false; // May or may not have been true
  }

  // Sanity check: wall observations should always be consistent
  uint64_t obs_check = MakeObservation(state);
  for (int i = 4; i <= 7; i++)
    assert(CheckFlag(obs, i) == CheckFlag(obs_check, i));

  // In the case where pocman hits a ghost, any food at pocman's position will
  // not be registered as eaten (see Pocman::Step()). We correct this error
  // before proceeding.
  int poc_index = maze_.Index(state.pocman_pos_);
  if (state.food_[poc_index] == 1)
    EatFoodInCurrentCell(state);

  // Which direction from pocman a ghost is placed in
  vector<int> dir_from_poc(num_ghosts_, -1);

  bool can_hear = CheckFlag(obs, 9);

  // In each direction, where to start and stop scanning
  int range_start = can_hear ? 1 : 1 + hear_range_;
  vector<int> range_end(4);

  // Whether pocman can hear any ghost we place
  bool placed_within_hearing = false;

  // Cells in which ghosts cannot be present
  set<Coord> blacklist = { state.pocman_pos_ };
  if (!can_hear) {
    for (int x = -hear_range_; x <= hear_range_; x++)
      for (int y = -hear_range_; y <= hear_range_; y++) {
        Coord pos = state.pocman_pos_ + Coord(x, y);
        if (Coord::ManhattanDistance(state.pocman_pos_, pos) <= hear_range_)
          blacklist.insert(pos);
      }
  }
  for (int d = 0; d < 4; d++) {
    if (!CheckFlag(obs, d)) {
      Coord nextpos = state.pocman_pos_ + Coord::Compass[d];
      while (maze_.Inside(nextpos) && Passable(nextpos)) {
        blacklist.insert(nextpos);
        nextpos += Coord::Compass[d];
      }
    }
  }

  list<int> free_pool; // The set G - S (see function doc above)
  for (int g = 0; g < num_ghosts_; g++)
    if (blacklist.find(state.ghost_pos_[g]) != blacklist.end())
      free_pool.push_back(g);

  // Keep ghosts closest to pocman in directions where pocman can see a ghost.
  // Put the ones beyond in the free pool.
  for (int d = 0; d < 4; d++) {
    if (CheckFlag(obs, d)) {
      range_end[d] = range_start; // one past the end
      Coord nextpos = state.pocman_pos_ + Coord::Compass[d] * range_start;
      int closest_ghost = -1;
      while (maze_.Inside(nextpos) && Passable(nextpos)) {
        for (int g = 0; g < num_ghosts_; g++) {
          if (state.ghost_pos_[g] != nextpos)
            continue;
          if (closest_ghost == -1) {
            closest_ghost = g;
            dir_from_poc[g] = d;
            if (Coord::ManhattanDistance(state.pocman_pos_, nextpos) <=
                hear_range_)
              placed_within_hearing = true;
          }
          else
            free_pool.push_back(g);
        }
        range_end[d]++;
        nextpos += Coord::Compass[d];
      }
      assert(range_end[d] > range_start);
    }
  }

  for (int g = 0; g < num_ghosts_; g++)
    if (dir_from_poc[g] == -1 && 
        find(free_pool.begin(), free_pool.end(), g) == free_pool.end())
      free_pool.push_back(g);

  // Satisfy remaining directions, if any, where pocman can see a ghost
  for (int d = 0; d < 4; d++) {
    if (CheckFlag(obs, d) &&
        find(dir_from_poc.begin(), dir_from_poc.end(), d) == dir_from_poc.end()) {
      int dist = rand_r(&seed) % (range_end[d] - range_start) + range_start;
      int g = free_pool.front();
      state.ghost_pos_[g] = state.pocman_pos_ + Coord::Compass[d] * dist;
      dir_from_poc[g] = d;
      state.ghost_dir_[g] = -1;
      if (Coord::ManhattanDistance(state.pocman_pos_, state.ghost_pos_[g]) <=
          hear_range_)
        placed_within_hearing = true;
      free_pool.pop_front();
    }
  }

  // Move ghosts in blacklisted cells to random cells
  for (auto g: free_pool) {
    if (blacklist.find(state.ghost_pos_[g]) == blacklist.end())
      continue;
    while (1) {
      int sx = maze_.GetXSize(), sy = maze_.GetYSize();
      Coord maybe(rand_r(&seed) % sx, rand_r(&seed) % sy);
      if (Passable(maybe) && blacklist.find(maybe) == blacklist.end()) {
        state.ghost_pos_[g] = maybe;
        state.ghost_dir_[g] = -1;
        if (Coord::ManhattanDistance(maybe, state.pocman_pos_) <= hear_range_)
          placed_within_hearing = true;
        break;
      }
    }
  }

  // With all ghosts placed, we may still have not placed one within hearing
  // range of pocman. If pocman can hear a ghost, fix this by picking a ghost
  // randomly; if it was placed in a direction in which pocman sees a ghost, 
  // move it closer. Otherwise move it to a random location within hearing 
  // range.
  if (can_hear && !placed_within_hearing) {
    int ghost = rand_r(&seed) % num_ghosts_;
    if (dir_from_poc[ghost] == -1) {
      while (1) {
        int px = rand_r(&seed) % (2 * hear_range_ + 1) - hear_range_;
        int py = rand_r(&seed) % (2 * hear_range_ + 1) - hear_range_;
        Coord pos = state.pocman_pos_ + Coord(px, py);
        if (Coord::ManhattanDistance(state.pocman_pos_, pos) > hear_range_)
          continue;
        if (maze_.Inside(pos) && Passable(pos) &&
            blacklist.find(pos) == blacklist.end()) {
          state.ghost_pos_[ghost] = pos;
          state.ghost_dir_[ghost] = -1;
          break;
        }
      }
    }
    else {
      int dist = rand_r(&seed) % hear_range_ + 1;
      state.ghost_pos_[ghost] = state.pocman_pos_ +
          Coord::Compass[dir_from_poc[ghost]] * dist;
      assert(Passable(state.ghost_pos_[ghost]));
    }
  }

  // Pocman can smell food but there isn't any within range
  if (CheckFlag(obs, 8) && !SmellFood(state)) {
    bool placed = false;
    do {
      for (int i = -smell_range_; i <= smell_range_; i++)
        for (int j = -smell_range_; j <= smell_range_; j++) {
          Coord pos = state.pocman_pos_ + Coord(i, j);
          if (pos != state.pocman_pos_ && maze_.Inside(pos) &&
              CheckFlag(maze_(pos), E_SEED) && !CheckFlag(maze_(pos), E_POWER)) {
            if (rand_r(&seed) < food_prob_ * RAND_MAX &&
                state.food_[maze_.Index(pos)] == 0) {
              state.food_[maze_.Index(pos)] = 1;
              state.num_food_++;
              placed = true;
            }
          }
        }
    } while (!placed);
  }

  // Pocman cannot smell food but there is some in range
  else if (!CheckFlag(obs, 8) && SmellFood(state)) {
    for (int i = -smell_range_; i <= smell_range_; i++)
      for (int j = -smell_range_; j <= smell_range_; j++) {
        Coord pos = state.pocman_pos_ + Coord(i, j);
        if (maze_.Inside(pos) && state.food_[maze_.Index(pos)] == 1) {
          state.num_food_--;
          state.food_[maze_.Index(pos)] = -1;
        }
      }
  }

  // If num_food is 0, correct it. (We didn't receive terminal observation,
  // so num_food != 0)
  if (state.num_food_ == 0) {
    set<Coord> valid_positions;
    for (int i = 0; i < maze_.GetXSize(); i++)
      for (int j = 0; j < maze_.GetYSize(); j++) {
        Coord pos(i, j);
        if (maze_.Inside(pos) && Passable(pos) &&
            state.food_[maze_.Index(pos)] == 0)
          valid_positions.insert(pos);
      }
    if (!CheckFlag(obs, 8)) {
      for (auto p: smellfood_table_[state.pocman_pos_.X][state.pocman_pos_.Y])
        valid_positions.erase(maze_.IndexToCoord(p));
    }
    assert(!valid_positions.empty());
    bool placed = false;
    do {
      for (auto p: valid_positions)
        if (rand_r(&seed) < food_prob_ * RAND_MAX) {
          state.food_[maze_.Index(p)] = 1;
          state.num_food_++;
          placed = true;
        }
    } while (!placed);
  }

  return state;
}

