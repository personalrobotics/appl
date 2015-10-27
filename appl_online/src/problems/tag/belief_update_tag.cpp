#include "problems/tag/belief_update_tag.h"

vector<Particle<TagState>*> ParticleFilterUpdate<TagState>::UpdateImpl(
    const vector<Particle<TagState>*>& particles,
    int N,
    int act,
    uint64_t obs) {
  vector<Particle<TagState>*> ans;

  // In POMCP's implementation of Tag, the agent does not know its initial
  // position (it starts with the uniform distribution over the states),
  // and knows it only after the first step. If we sample the starting states 
  // from a uniform belief, and resample after the first step, we will lose 
  // many states - those inconsistent with the observable position of the 
  // robot. So for the first update, we sample from the entire state space 
  // instead of the particles we have.
  if (this->num_updates_ > 0) {
    // Step forward all particles
    for (auto p: particles) {
      double reward;
      double random_num = (double)rand_r(&(this->belief_update_seed_)) 
                          / RAND_MAX;
      Particle<TagState>* new_particle = this->model_.Copy(p);
      this->model_.Step(new_particle->state, random_num, act, reward);
      double obs_prob = this->model_.ObsProb(obs, new_particle->state, act);
      if (obs_prob) {
        new_particle->wt = p->wt * obs_prob;
        ans.push_back(new_particle);
      }
      else
        this->model_.Free(new_particle);
    }
  }
  this->Normalize(ans);

  if (ans.empty()) {
    // No resulting state is consistent with the given observation. Either we
    // we went completely off track, or this is first update and the step phase
    // above was not executed. So loop over all states and compute their probs.
    for (int s = 0; s < this->model_.NumStates(); s++) {
      double obs_prob = this->model_.ObsProb(obs, s, act);
      if (obs_prob) {
        Particle<TagState>* new_particle = this->model_.Allocate();
        new_particle->state = s;
        new_particle->id = 0; // dummy; will be recomputed after sampling below
        new_particle->wt = obs_prob;
        ans.push_back(new_particle);
      }
    }
    this->Normalize(ans);
    auto resampled_ans = this->Sample(ans, N);
    for (auto it: ans)
      this->model_.Free(it);
    return resampled_ans;
  }

  // Remove all particles below the threshold weight
  auto new_last_ptr = ans.begin();
  for (auto& p: ans)
    if (p->wt >= this->PARTICLE_WT_THRESHOLD) {
      swap(p, *new_last_ptr);
      ++new_last_ptr; 
    }
  if (new_last_ptr != ans.end()) {
    for (auto it = new_last_ptr; it != ans.end(); it++)
      this->model_.Free(*it);
    ans.erase(new_last_ptr, ans.end());
    this->Normalize(ans);
  }

  // Resample if we have < N particles or the no. of effective particles drops
  // below the threshold
  double num_eff_particles = 0;
  for (auto it: ans)
    num_eff_particles += it->wt * it->wt;
  num_eff_particles = 1 / num_eff_particles;
  if (num_eff_particles < N * 0.05 || ans.size() < N) {
    auto resampled_ans = this->Sample(ans, N);
    for (auto it: ans)
      this->model_.Free(it);
    ans = resampled_ans;
  } 

  return ans;
}
