#ifndef POCMAN_H
#define POCMAN_H

#include "globals.h"
#include "history.h"
#include "problems/pocman/coord.h"
#include "problems/pocman/grid.h"
#include "upper_bound/upper_bound.h"

class PocmanState : MemoryObject {
 public:
  PocmanState() : terminal_(false) {}

  // Dummy methods to satisfy implicit conversion requirements for
  // certain modules, so that the program can compile cleanly. 
  // These modules are never actually used with Pocman.
  PocmanState(int id) {}
  operator int() const { return 0; }

  Coord pocman_pos_;
  std::vector<Coord> ghost_pos_;
  std::vector<int> ghost_dir_;
  std::vector<int> food_; // -1 if eaten, 0 if not present, 1 if present
  int num_food_;
  int power_steps_;
  bool terminal_;
};

// Dummy hasher, like above
namespace std {
  template<>
  struct hash<PocmanState> {
    inline size_t operator()(const PocmanState& s) const {
      return s;
    }
  };
}

class Pocman : public Model<PocmanState>, public IUpperBound<PocmanState> {
 public:
  Pocman(unsigned initial_state_seed, const RandomStreams& streams);

  void Step(PocmanState& state, double randomNum, int action, double& reward,
            uint64_t& obs) const;

  void Step(PocmanState& state, double randomNum, int action, double& reward)
      const {
    uint64_t obs;
    Step(state, randomNum, action, reward, obs);
  }

  double ObsProb(uint64_t z, const PocmanState& state, int action) const;

  PocmanState GetStartState() const;

  PocmanState MakeConsistent(const PocmanState& state, uint64_t obs,
                             unsigned& seed) const;

  vector<pair<PocmanState, double>> InitialBelief() const;

  void PrintState(const PocmanState& state, ostream& out = cout) const;

  void PrintObs(uint64_t obs, ostream& out = cout) const;

  int NumActions() const { return 4; }

  bool IsTerminal(const PocmanState& s) const { return s.terminal_; }

  uint64_t TerminalObs() const { return 1 << 10; }

  int DefaultActionPreferred(const History& history, const PocmanState& state)
      const;

  double UpperBound(History& history,
                    const vector<Particle<PocmanState>*>& particles,
                    int stream_position, const Model<PocmanState>& model) const;

  double FringeLowerBound(const vector<Particle<PocmanState>*>& particles)
      const {
    return reward_default_ * particles[0]->state.num_food_ +
           reward_die_;
  };

  Particle<PocmanState>* Copy(const Particle<PocmanState>* particle) const {
    Particle<PocmanState>* new_particle = Allocate();
    *new_particle = *particle;
    return new_particle;
  }

  Particle<PocmanState>* Allocate() const {
    return memory_pool_.Allocate();
  }

  void Free(Particle<PocmanState>* particle) const {
    memory_pool_.Free(particle);
  }

 private:
  void InitMazeMicro();
  void InitMazeMini();
  void InitMazeFull();
  void InitLookupTables();
  PocmanState GetStartState(unsigned seed) const;
  void MoveGhost(PocmanState& state, int g, double rand_num) const;
  void MoveGhostAggressive(PocmanState& state, int g, double rand_num) const;
  void MoveGhostDefensive(PocmanState& state, int g, double rand_num) const;
  void MoveGhostRandom(PocmanState& state, int g, double rand_num) const;
  void NewLevel(PocmanState& state, unsigned& seed) const;
  int SeeGhost(const PocmanState& state, int dir) const;
  bool HearGhost(const PocmanState& state) const;
  bool SmellFood(const PocmanState& state) const;
  void EatFoodInCurrentCell(PocmanState& state) const;
  bool Passable(const Coord& pos) const {
    return CheckFlag(maze_(pos), E_PASSABLE);
  }
  Coord NextPos(const Coord& from, int dir) const {
    return nextpos_table_[from.X][from.Y][dir];
  }
  void SetWallObs(int& obs, const PocmanState& state) const {
    obs |= wall_obs_table_[state.pocman_pos_.X][state.pocman_pos_.Y];
  }
  uint64_t MakeObservation(const PocmanState& state) const;
  bool CheckFlag(int flags, int bit) const { return (flags & (1 << bit)) != 0; }
  void SetFlag(int& flags, int bit) const { flags = (flags | (1 << bit)); }

  enum {
    E_PASSABLE,
    E_SEED,
    E_POWER
  };

  unsigned initial_state_seed_;
  Grid<int> maze_;
  int num_ghosts_, passage_y_, ghost_range_, smell_range_, hear_range_;
  Coord pocman_home_, ghost_home_;
  double food_prob_, chase_prob_, defensive_slip_;
  double reward_clear_level_, reward_default_, reward_die_;
  double reward_eat_food_, reward_eat_ghost_, reward_hit_wall_;
  int power_num_steps_;

  // Various lookup tables for efficiency
  vector<vector<vector<Coord>>> nextpos_table_;
  vector<vector<vector<vector<int>>>> relative_dir_table_;
  vector<vector<vector<int>>> smellfood_table_;
  vector<vector<vector<vector<pair<int, Coord>>>>> moveghost_random_table_;
  vector<vector<int>> preferred_powerpill_table_;
  vector<vector<vector<vector<int>>>> preferred_normal_table_;
  vector<vector<int>> wall_obs_table_;

  mutable MemoryPool<Particle<PocmanState>> memory_pool_;
};

#endif
