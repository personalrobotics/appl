#ifndef TAG_PARTICLE_FILTER_UPDATE
#define TAG_PARTICLE_FILTER_UPDATE

#include "belief_update/belief_update_particle.h"
#include "problems/tag/tag.h"

/* This class specializes the particle filter for Tag. It is almost identical
 * to the generic particle filter except for a minor change explained in the
 * implementation.
 */
template<>
class ParticleFilterUpdate<TagState> : public BeliefUpdate<TagState> {
 public:
  ParticleFilterUpdate(unsigned belief_update_seed, 
                       const Model<TagState>& model) 
      : BeliefUpdate<TagState>(belief_update_seed, model)
  {}

 private:
  vector<Particle<TagState>*> UpdateImpl(
      const vector<Particle<TagState>*>& particles,
      int N,
      int act,
      uint64_t obs);
};

#endif
