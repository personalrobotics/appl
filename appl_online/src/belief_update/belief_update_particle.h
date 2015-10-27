#ifndef PARTICLE_FILTER_UPDATE
#define PARTICLE_FILTER_UPDATE

#include "belief_update/belief_update.h"

/* This class implements sequential importance resampling (SIR) of particles.
 * Its time and space complexity are linear in the number of particles.
 */
template<typename T>
class ParticleFilterUpdate : public BeliefUpdate<T> {
 public:
  ParticleFilterUpdate(unsigned belief_update_seed, const Model<T>& model) 
      : BeliefUpdate<T>(belief_update_seed, model)
  {}

 protected:
  static constexpr double NUM_EFF_PARTICLE_FRACTION = 0.05;

 private:
  vector<Particle<T>*> UpdateImpl(
      const vector<Particle<T>*>& particles,
      int N,
      int act,
      uint64_t obs);
};

template<typename T>
constexpr double ParticleFilterUpdate<T>::NUM_EFF_PARTICLE_FRACTION;

template<typename T>
vector<Particle<T>*> ParticleFilterUpdate<T>::UpdateImpl(
    const vector<Particle<T>*>& particles,
    int N,
    int act,
    uint64_t obs) {

  vector<Particle<T>*> ans;
  double reward;

  // Step forward all particles
  for (auto p: particles) {
    double random_num = (double)rand_r(&(this->belief_update_seed_)) / RAND_MAX;
    Particle<T>* new_particle = this->model_.Copy(p);
    this->model_.Step(new_particle->state, random_num, act, reward);
    double obs_prob = this->model_.ObsProb(obs, new_particle->state, act);
    if (obs_prob) {
      new_particle->wt = p->wt * obs_prob;
      ans.push_back(new_particle);
    }
    else
      this->model_.Free(new_particle);
  }
  this->Normalize(ans);

  if (ans.empty()) {
    // No resulting state is consistent with the given observation, so create
    // states randomly until we have enough that are consistent.
    cerr << "WARNING: Particle filter empty. Bootstrapping with random states"
         << endl;
    int num_sampled = 0;
    while (num_sampled < N) {
      T s = this->model_.RandomState(this->belief_update_seed_, obs);
      double obs_prob = this->model_.ObsProb(obs, s, act);
      if (obs_prob) {
        Particle<T>* new_particle = this->model_.Allocate();
        new_particle->state = s;
        new_particle->id = num_sampled++;
        new_particle->wt = obs_prob;
        ans.push_back(new_particle);
      }
    }
    this->Normalize(ans);
    return ans;
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

  // Resample if we have < N particles or no. of effective particles drops 
  // below the threshold
  double num_eff_particles = 0;
  for (auto it: ans)
    num_eff_particles += it->wt * it->wt;
  num_eff_particles = 1 / num_eff_particles;
  if (num_eff_particles < N * NUM_EFF_PARTICLE_FRACTION || ans.size() < N) {
    auto resampled_ans = this->Sample(ans, N);
    for (auto it: ans)
      this->model_.Free(it);
    ans = resampled_ans;
  } 

  return ans;
}

#endif
