#ifndef TAG_PARTICLE_FILTER_UPDATE
#define TAG_PARTICLE_FILTER_UPDATE

#include "belief_update/belief_update_particle.h"
#include "problems/pocman/pocman.h"

/* This class specializes the particle filter for Pocman.
 */
template<>
class ParticleFilterUpdate<PocmanState> : public BeliefUpdate<PocmanState> {
 public:
  ParticleFilterUpdate(unsigned belief_update_seed, 
                       const Model<PocmanState>& model) 
      : BeliefUpdate<PocmanState>(belief_update_seed, model)
  {}

 private:
  vector<Particle<PocmanState>*> UpdateImpl(
      const vector<Particle<PocmanState>*>& particles,
      int N,
      int act,
      uint64_t obs);
};

vector<Particle<PocmanState>*> ParticleFilterUpdate<PocmanState>::UpdateImpl(
    const vector<Particle<PocmanState>*>& particles,
    int N,
    int act,
    uint64_t obs) {

  vector<Particle<PocmanState>*> ans;
  double reward;

  // Step forward all particles
  for (int i = 0; i < particles.size(); i++) {
    auto p = particles[i];
    double random_num = (double)rand_r(&(this->belief_update_seed_)) / RAND_MAX;
    Particle<PocmanState>* new_particle = this->model_.Copy(p);
    this->model_.Step(new_particle->state, random_num, act, reward);
    double obs_prob = this->model_.ObsProb(obs, new_particle->state, act);
    if (obs_prob)
      ans.push_back(new_particle);
    else {
      Pocman* model = (Pocman*)(&(this->model_));
      PocmanState s = model->MakeConsistent(new_particle->state, obs,
                                            this->belief_update_seed_);
      assert(this->model_.ObsProb(obs, s, act));
      Particle<PocmanState>* new_particle2 = this->model_.Allocate();
      new_particle2->state = s;
      new_particle2->id = i;
      new_particle2->wt = 1.0 / Globals::config.n_particles;
      ans.push_back(new_particle2);
      this->model_.Free(new_particle);
    }
  }

  return ans;
}

#endif
