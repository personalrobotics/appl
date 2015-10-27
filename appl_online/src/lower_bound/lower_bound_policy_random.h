#ifndef LOWER_BOUND_POLICY_RANDOM_H
#define LOWER_BOUND_POLICY_RANDOM_H

#include "lower_bound/lower_bound_policy.h"

/* This class refines PolicyLowerBound to use a random action as the best
 * action for each node. If the knowledge level is 2, a preferred action 
 * is used, whereas if it is 1, a legal action is used. The computation 
 * of preferred and legal actions for a given history is delegated to the
 * model. The knowledge level can be set using a command line parameter to 
 * the program.
 *
 * Space complexity: O(no. of actions generated for the state)
 * Time complexity (per query): O(no. of particles * search_depth)
 */
template<typename T>
class RandomPolicyLowerBound : public PolicyLowerBound<T> {
 public:
  RandomPolicyLowerBound(const RandomStreams& streams, 
                         int knowledge)
      : PolicyLowerBound<T>(streams),
        knowledge_(knowledge)
  {}

  int Action(const History& history,
             const vector<Particle<T>*>& particles,
             const Model<T>& model) const;

 private:
  int knowledge_;
};

template<typename T>
int RandomPolicyLowerBound<T>::Action(
    const History& history,
    const vector<Particle<T>*>& particles,
    const Model<T>& model) const {
  const T& state = particles[0]->state;

  if (knowledge_ >= 2) {
    int act = model.DefaultActionPreferred(history, state);
    if (act != -1)
      return act;
  }

  return model.DefaultActionLegal(history, state);
}

#endif
