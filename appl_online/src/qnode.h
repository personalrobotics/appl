#ifndef QNODE_H
#define QNODE_H

#include "globals.h"
#include "history.h"
#include "lower_bound/lower_bound.h"
#include "model.h"
#include "upper_bound/upper_bound.h"
#include "vnode.h"

/* This class encapsulates an AND-node (Q-node) of the belief tree, branching
 * on observations. It maps the set of observations seen during simulations to
 * the VNodes that they reach at the next level, and also maintains some 
 * bookkeeping information.
 */
template<typename T>
class QNode {
 public:
  // Params:
  // @obs_to_particles: a mapping from observations seen to the particles (after
  // the transition) that generated the observations. The particles for each
  // observation become the representative set for the corresponding v-node
  // at the next level.
  // @depth: depth of the v-node *above* this node.
  // @action: action that led to this q-node.
  // @first_step_reward: The average first step reward of particles when they
  // took action @action.
  // @history: history upto the v-node *above* this node.
  // @debug: Flag controlling debugging output.
  QNode(MAP<uint64_t, vector<Particle<T>*>>& obs_to_particles,
        int depth,
        int action,
        double first_step_reward,
        History& history,
        const Model<T>& model,
        const ILowerBound<T>& lb,
        const IUpperBound<T>& ub,
        bool debug=false);

  double UpperBound() const;

  double LowerBound() const;

  double Prune(int& total_pruned) const;

  // Returns the observation with the highest weighted excess uncertainty
  // ("WEU"), along with the value of the WEU.
  // @root: Root of the search tree, passed to facilitate computation of the 
  //        excess uncertainty
  pair<uint64_t, double> BestWEUO(const unique_ptr<VNode<T>>& root) const;

  // Returns the v-node corresponding to a given observation
  unique_ptr<VNode<T>>& Belief(uint64_t obs) {
    return obs_to_node_[obs];
  }

  double first_step_reward() const { return first_step_reward_; }

 private:
  int depth_; // Depth of the v-node *above* this node
  double weight_sum_; // Combined weight of particles at this node
  double first_step_reward_;
  MAP<uint64_t, unique_ptr<VNode<T>>> obs_to_node_; // Map from obs to v-node
};

template<typename T>
QNode<T>::QNode(
    MAP<uint64_t, vector<Particle<T>*>>& obs_to_particles, int depth, 
    int action, double first_step_reward, History& history, 
    const Model<T>& model, const ILowerBound<T>& lb, const IUpperBound<T>& ub,
    bool debug)
    : depth_(depth),
      first_step_reward_(first_step_reward) {
  weight_sum_ = 0;
  for (auto& r: obs_to_particles) {
    double obs_ws = 0;
    for (auto p: r.second)
      obs_ws += p->wt;
    weight_sum_ += obs_ws;

    history.Add(action, r.first);
    double l = lb.LowerBound(history, r.second, depth + 1, model).first;
    double u = ub.UpperBound(history, r.second, depth + 1, model);
    history.RemoveLast();
    obs_to_node_[r.first] = unique_ptr<VNode<T>>(
      new VNode<T>(std::move(r.second), l, u, depth + 1, obs_ws, false));
  }
}

template<typename T>
double QNode<T>::UpperBound() const {
  double ans = 0;
  for (auto& it: obs_to_node_)
    ans += it.second->ubound() * it.second->weight() / weight_sum_;
  return ans;
}

template<typename T>
double QNode<T>::LowerBound() const {
  double ans = 0;
  for (auto& it: obs_to_node_)
    ans += it.second->lbound() * it.second->weight() / weight_sum_;
  return ans;
}

template<typename T>
double QNode<T>::Prune(int& total_pruned) const {
  double cost = 0;
  for (auto& it: obs_to_node_)
    cost += it.second->Prune(total_pruned);
  return cost;
}

template<typename T>
pair<uint64_t, double> QNode<T>::BestWEUO(
    const unique_ptr<VNode<T>>& root) const {
  double weighted_eu_star = -Globals::INF;
  uint64_t o_star = 0;
  for (auto& it: obs_to_node_) {
    double weighted_eu = it.second->weight() / weight_sum_ *
                         Globals::ExcessUncertainty(
                           it.second->lbound(), it.second->ubound(),
                           root->lbound(), root->ubound(), depth_ + 1);
    if (weighted_eu > weighted_eu_star) {
      weighted_eu_star = weighted_eu;
      o_star = it.first;
    }
  }
  return {o_star, weighted_eu_star};
}

#endif
