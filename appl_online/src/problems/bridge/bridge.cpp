#include "bridge.h"

Bridge::Bridge(unsigned initial_state_seed) {
  man_position_ = rand_r(&initial_state_seed) % 11;
}

void Bridge::Step(BridgeState& s, double random_num, int action, double& reward,
                  uint64_t& obs) const {
  obs = 1;
  if (action == LEFT) {
    reward = -1;
    if (s > 1)
      s = s - 1;
  } 
  else if (action == RIGHT) {
    if (s < BRIDGELENGTH) {
      reward = -1;
      s = s + 1;
    }
    else {
      reward = 0;
      obs = TerminalObs();
      s = 0;
    }
  }
  else { // HELP
    reward = -20 - s;
    obs = TerminalObs();
    s = 0;
  }
}

double Bridge::ObsProb(uint64_t obs, const BridgeState& s, int a) const {
  return s == 0 ? (obs == TerminalObs()) : (obs == 1);
}

int Bridge::DefaultActionPreferred(const History& history,
    const BridgeState& state) const {
  return HELP;
}

vector<pair<BridgeState, double>> Bridge::InitialBelief() const {
  vector<pair<BridgeState, double>> belief;
  int leftend = man_position_ - 1, rightend = man_position_ + 1;
  if (leftend < 1) leftend = 1;
  if (rightend > BRIDGELENGTH) rightend = BRIDGELENGTH;
  for (int pos = leftend; pos <= rightend; pos++)
    belief.push_back({ pos, 1.0 / (rightend - leftend + 1) });
  return belief;
}

void Bridge::PrintState(const BridgeState& state, ostream& out) const {
  out << "Man at " << state << endl;
}

