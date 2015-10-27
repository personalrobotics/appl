#ifndef BRIDGE_H
#define BRIDGE_H

#include "globals.h"
#include "model.h"
#include "memorypool.h"
#include "lower_bound/lower_bound_policy_random.h"
#include "upper_bound/upper_bound_stochastic.h"

/* Problem Description
 *
 * Consider being on a ledge between A and B where the distance between A
 * and B is 10 steps. You can walk towards A or B where each step costs -1.
 * If you reach B, you can get off and earn a reward of 0. You cannot get
 * off at A by yourself, but at any time, you can call for help which will
 * appear at A and guide you off immediately at a cost of (-20 - distance
 * from A). To make it partially observable, you only know your true position
 * within one step and there is no observation. The default policy calls
 * for help.
 */
class BridgeState {
 public:
  BridgeState() : id_(0) {}

  BridgeState(int id) : id_(id) {}

  operator int() const { return id_; }

 private:
  int id_;
};

namespace std {
  template<>
  struct hash<BridgeState> {
    inline size_t operator()(const BridgeState& s) const {
      return s;
    }
  };
}

class Bridge : public Model<BridgeState> {
 public:
  Bridge(unsigned initial_state_seed);

  void Step(BridgeState& s, double random_num, int action, 
            double& reward, uint64_t& obs) const;

  void Step(BridgeState& s, double random_num, int action, double& reward)
      const {
    uint64_t obs;
    Step(s, random_num, action, reward, obs);
  }

  double ObsProb(uint64_t obs, const BridgeState& s, int a) const;

  double FringeUpperBound(const BridgeState& s) const {
    return 0;
  }

  double FringeLowerBound(const vector<Particle<BridgeState>*>& particles) 
      const {
    return Globals::config.discount < 1 
           ? -1 / (1 - Globals::config.discount) 
           : -100;
  }

  int DefaultActionPreferred(const History& history, const BridgeState& state) 
      const;

  BridgeState GetStartState() const { return man_position_; }

  BridgeState RandomState(unsigned& seed, uint64_t obs) const {
    return rand_r(&seed) % NumStates();
  }

  vector<pair<BridgeState, double>> InitialBelief() const;

  void PrintState(const BridgeState& state, ostream& out = cout) const;

  void PrintObs(uint64_t obs, ostream& out = cout) const {};

  int NumStates() const { return BRIDGELENGTH + 1; }

  int NumActions() const { return 3; }

  bool IsTerminal(const BridgeState& s) const { return s == 0; }

  uint64_t TerminalObs() const { return TERMINAL_OBSERVATION; }

  Particle<BridgeState>* Copy(const Particle<BridgeState>* particle) const {
    Particle<BridgeState>* new_particle = Allocate();
    *new_particle = *particle;
    return new_particle;
  }

  Particle<BridgeState>* Allocate() const {
    return memory_pool_.Allocate();
  }

  void Free(Particle<BridgeState>* particle) const {
    memory_pool_.Free(particle);
  }

 private:
  int man_position_;

  static constexpr int LEFT = 0;
  static constexpr int RIGHT = 1;
  static constexpr int HELP = 2;
  static constexpr int BRIDGELENGTH = 10;
  static constexpr int TERMINAL_OBSERVATION = 0;

  mutable MemoryPool<Particle<BridgeState>> memory_pool_;
};

#endif
