#ifndef WORLD_H
#define WORLD_H

#include "globals.h"
#include "model.h"

/* This class maintains the current state of the world and steps it forward
 * whenever the agent takes an action and receives an observation.
 */
template<typename T>
class World {
 public:
  World(unsigned seed, const Model<T>& model)
    : state_(model.GetStartState()),
      initial_state_(state_),
      seed_(seed),
      initial_seed_(seed),
      model_(model)
  {}

  double UndiscountedReturn() const;

  double DiscountedReturn() const;

  // Advances the current state of the world.
  void Step(int action, uint64_t& obs, double& reward);

  // Resets the world to have the same initial state and seed so that
  // a sequence of updates can be reproduced exactly.
  void Reset();

 private:
  T state_;
  const T initial_state_;
  unsigned seed_;
  const unsigned initial_seed_;
  const Model<T>& model_;
  vector<double> rewards_;
};

template<typename T>
void World<T>::Reset() {
  state_ = initial_state_;
  seed_ = initial_seed_;
  rewards_.clear();
}

template<typename T>
double World<T>::UndiscountedReturn() const {
  return accumulate(rewards_.begin(), rewards_.end(), 0);
}

template<typename T>
double World<T>::DiscountedReturn() const {
  double ans = 0, multiplier = 1;
  for (auto r: rewards_) {
    ans += multiplier * r;
    multiplier *= Globals::config.discount;
  }
  return ans;
}

template<typename T>
void World<T>::Step(int action, uint64_t& obs, double& reward) {
  double random_num = (double)rand_r(&seed_) / RAND_MAX;
  model_.Step(state_, random_num, action, reward, obs);
  cout << "Action = " << action << endl;
  cout << "State = \n"; model_.PrintState(state_);
  cout << "Observation: "; model_.PrintObs(obs); cout << endl;
  cout << "Reward = " << reward << endl;
  rewards_.push_back(reward);
}

#endif
