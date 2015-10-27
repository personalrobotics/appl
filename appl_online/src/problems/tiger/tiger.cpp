#include "tiger.h"

Tiger::Tiger(unsigned initial_state_seed) {
  tiger_position_ = rand_r(&initial_state_seed) % 2;
}

void Tiger::Step(TigerState& s, double random_num, int action, double& reward)
    const {
  if (action == LEFT || action == RIGHT) {
    reward = s != action ? 10 : -100;
    s = random_num <= 0.5 ? LEFT : RIGHT;
  } 
  else
    reward = -1;
}

void Tiger::Step(TigerState& s, double random_num, int action, double& reward, 
                 uint64_t& obs) const {
  if (action == LEFT || action == RIGHT) {
    reward = s != action ? 10 : -100;
    obs = 2;
    s = random_num <= 0.5 ? LEFT: RIGHT;
  } 
  else {
    reward = -1;
    if (random_num <= 0.85)
      obs = s;
    else
      obs = (1 - s);
  }
}

double Tiger::ObsProb(uint64_t obs, const TigerState& s, int a) const {
  if (a != LISTEN) return 0.5;

  return s == obs ? 0.85 : 0.15;
}

double Tiger::FringeUpperBound(const TigerState& s) const {
  return 10;
}

int Tiger::DefaultActionPreferred(const History& history,
    const TigerState& state) const {
  if (history.Size() == 0)
    return -1;

  int count_diff = 0;
  for (int i = history.Size() - 1; i >= 0 && history.Action(i) == LISTEN; i--)
    count_diff += history.Observation(i) == LEFT ? 1 : -1;
  if (count_diff >= 2)
    return LEFT;
  else if (count_diff <= -2)
    return RIGHT;

  return LISTEN;
}

double Tiger::FringeLowerBound(const vector<Particle<TigerState>*>& particles) 
    const {
  return Globals::config.discount < 1 
         ? -1 / (1 - Globals::config.discount) 
         : -100;
}

vector<pair<TigerState, double>> Tiger::InitialBelief() const {
  return {{LEFT, 0.5}, {RIGHT, 0.5}};
}

void Tiger::PrintState(const TigerState& state, ostream& out) const {
  out << (state == LEFT ? "LEFT" : "RIGHT") << endl;
}

