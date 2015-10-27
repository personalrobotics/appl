#ifndef POLICY_LOWER_BOUND_H
#define POLICY_LOWER_BOUND_H

#include "lower_bound/lower_bound.h"

/* This class computes the the lower bound of a node, represented by its
 * history, by executing a policy tree on the set of particles at that node.
 * After an action is taken, the particles are grouped by the observations
 * generated, and the simulation continues recursively at the next level until
 * the particles reach a terminal state or maximum depth. The best action to 
 * take at each step can be specified via a virtual method by refining this
 * class. Note that the action for a given history must be fixed.
 */
template<typename T>
class PolicyLowerBound : public ILowerBound<T> {
 public:
  PolicyLowerBound(const RandomStreams& streams) : ILowerBound<T>(streams) {}

  virtual ~PolicyLowerBound() {}

  pair<double, int> LowerBound(History& history,
                               const vector<Particle<T>*>& particles,
                               int stream_position,
                               const Model<T>& model) const;

 protected:
  // Returns the best action for a set of particles. Define this in 
  // implementations of this class.
  virtual int Action(const History& history,
                     const vector<Particle<T>*>& particles,
                     const Model<T>& model) const = 0;

 private:
  pair<double, int> LowerBoundImpl(History& history,
                                   vector<Particle<T>*>& particles,
                                   int stream_position,
                                   const Model<T>& model) const;
};

template<typename T>
pair<double, int> PolicyLowerBound<T>::LowerBound(
    History& history,
    const vector<Particle<T>*>& particles,
    int stream_position,
    const Model<T>& model) const {
  // Copy the particles so that we can modify them
  auto copy = particles;
  for (auto& it: copy)
    it = model.Copy(it);
  pair<double, int> lb = LowerBoundImpl(history, copy, stream_position, model);
  for (auto it: copy)
    model.Free(it);
  return lb;
}

template<typename T>
pair<double, int> PolicyLowerBound<T>::LowerBoundImpl(
    History& history,
    vector<Particle<T>*>& particles,
    int stream_position,
    const Model<T>& model) const {

  bool debug = false;
  if (debug) { cerr << "Lower bound depth = " << stream_position << endl; }

  // Terminal states have a unique observation, so if we took that branch it
  // means that all particles must be in a terminal state.
  if (model.IsTerminal(particles[0]->state))
    return {0, -1};

  if (stream_position >= Globals::config.search_depth)
    // The actual action should never be required when 
    // depth >= config.search_depth, so we return a dummy value.
    return { model.FringeLowerBound(particles), -1/*dummy*/ };

  int act = Action(history, particles, model);
  if (debug) { cerr << "act = " << act << endl; }

  // Compute value based on one-step lookahead
  double weight_sum = 0;
  MAP<uint64_t, pair<double, vector<Particle<T>*>>> partitioned_particles;
  double first_step_reward = 0;
  for (auto& p: particles) {
    uint64_t o; double r;
    model.Step(p->state, this->streams_.Entry(p->id, stream_position), 
               act, r, o);
    auto& ref = partitioned_particles[o];
    ref.first += p->wt;
    ref.second.push_back(p);
    first_step_reward += r * p->wt;
    weight_sum += p->wt;
  }
  first_step_reward /= weight_sum;

  double remaining_reward = 0;
  for (auto& it: partitioned_particles) {
    history.Add(act, it.first);
    remaining_reward += Globals::config.discount *
                        LowerBoundImpl(history, it.second.second, 
                                       stream_position + 1, model).first *
                        it.second.first / weight_sum;
    history.RemoveLast();
  }

  return { first_step_reward + remaining_reward, act };
}

#endif
