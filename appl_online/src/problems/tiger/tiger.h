#ifndef TIGER_H
#define TIGER_H

#include "globals.h"
#include "model.h"
#include "solver.h"
#include "lower_bound/lower_bound_policy_random.h"
#include "upper_bound/upper_bound_stochastic.h"

class TigerState {
 public:
  TigerState() : id_(0) {}

  TigerState(int id) : id_(id) {}

  operator int() const { return id_; }

 private:
  int id_;
};

namespace std {
  template<>
  struct hash<TigerState> {
    inline size_t operator()(const TigerState& s) const {
      return s;
    }
  };
}

class Tiger : public Model<TigerState> { 
 public:
  Tiger(unsigned initial_state_seed);

  void Step(TigerState& s, double random_num, int action, double& reward, 
            uint64_t& obs) const;

  void Step(TigerState& s, double random_num, int action, double& reward) const;

  double ObsProb(uint64_t obs, const TigerState& s, int a) const;

  double FringeUpperBound(const TigerState& s) const;

  double FringeLowerBound(const vector<Particle<TigerState>*>& particles) const;

  int DefaultActionPreferred(const History& history, const TigerState& state) 
      const;

  TigerState GetStartState() const { return tiger_position_; }

  TigerState RandomState(unsigned& seed, uint64_t obs) const {
    return rand_r(&seed) % NumStates();
  }

  vector<pair<TigerState, double>> InitialBelief() const;

  void PrintState(const TigerState& state, ostream& out = cout) const;

  void PrintObs(uint64_t obs, ostream& out = cout) const {};

  int NumStates() const { return 2; }

  int NumActions() const { return 3; }

  bool IsTerminal(const TigerState& s) const { return false; }

  uint64_t TerminalObs() const { return 100; } // never encountered

  Particle<TigerState>* Copy(const Particle<TigerState>* particle) const {
    Particle<TigerState>* new_particle = Allocate();
    *new_particle = *particle;
    return new_particle;
  }

  Particle<TigerState>* Allocate() const {
    return memory_pool_.Allocate();
  }

  void Free(Particle<TigerState>* particle) const {
    memory_pool_.Free(particle);
  }

 private:
  int tiger_position_;

  // LEFT and RIGHT are used for states, actions and observations; LISTEN is
  // for actions.
  enum {
    LEFT,
    RIGHT,
    LISTEN
  };

  mutable MemoryPool<Particle<TigerState>> memory_pool_;
};

#endif
