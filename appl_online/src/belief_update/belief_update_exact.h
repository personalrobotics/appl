#ifndef EXACT_BELIEF_UPDATE_H
#define EXACT_BELIEF_UPDATE_H

#include "belief_update/belief_update.h"

/* This class implements exact belief updates (probability distribution over
 * states). The time and space complexity for an update are O(S^2) and O(S),
 * so use it only for small problems. The problem state should define implicit
 * conversions to int to enable this module to index internal data structures
 * with state.
 */
template<typename T>
class ExactBeliefUpdate : public BeliefUpdate<T> {
 public:
  ExactBeliefUpdate(unsigned belief_update_seed, const Model<T>& model)
      : BeliefUpdate<T>(belief_update_seed, model),
        T_(model.TransitionMatrix()) {
    auto init_belief = model.InitialBelief();
    belief_ = decltype(belief_)(init_belief.begin(), init_belief.end());
  }

 private:
  vector<Particle<T>*> UpdateImpl(
      const vector<Particle<T>*>& particles,
      int N,
      int act,
      uint64_t obs);

  UMAP<T, double> belief_; // Mapping from state to probability.
  vector<vector<UMAP<T, double>>> T_;
};

template<typename T>
vector<Particle<T>*> ExactBeliefUpdate<T>::UpdateImpl(
    const vector<Particle<T>*>& particles,
    int N, 
    int act, 
    uint64_t obs) {
  // Update equation:
  // b'(s') = { O(s',a,z) \sum_{s \in S} T(s,a,s')b(s) } / norm. constant

  int S = this->model_.NumStates();
  int A = this->model_.NumActions();
  assert(T_.size() == S);
  UMAP<T, double> new_belief;

  for (auto& it: belief_) {
    int s = it.first;
    assert(T_[s].size() == A);
    for (auto& it2: T_[s][act]) {
      auto& s_ = it2.first;
      new_belief[s_] += it2.second * it.second;
    }
  }
  double prob_sum = 0;
  for (auto& it: new_belief) {
    it.second *= this->model_.ObsProb(obs, it.first, act);
    prob_sum += it.second;
  }
  double prob_sum2 = 0;
  for (auto it = new_belief.begin(); it != new_belief.end(); ) {
    it->second /= prob_sum;
    if (it->second < this->PARTICLE_WT_THRESHOLD)
      it = new_belief.erase(it);
    else {
      prob_sum2 += it->second;
      it++;
    }
  }
  belief_.swap(new_belief);
  vector<Particle<T>*> pool;
  for (auto& it: belief_) {
    it.second /= prob_sum2;
    auto new_particle = this->model_.Allocate();
    new_particle->state = it.first;
    new_particle->id = 0; // dummy; will be recomputed after sampling below
    new_particle->wt = it.second;
    pool.push_back(new_particle);
  }

  auto ans = this->Sample(pool, N);
  for (auto it: pool)
    this->model_.Free(it);
  return ans;
}

#endif
