#ifndef MODEL_H
#define MODEL_H

#include "globals.h"
#include "history.h"

class RandomStreams;
template<typename T> class Particle;
template<typename T> class ILowerBound;
template<typename T> class IUpperBound;
template<typename T> class BeliefUpdate;

/* This template forms the superclass for a problem specification. 
 * To implement a new problem, define a class X representing a state in the
 * problem, and a class that inherits and implements this template 
 * parameterized with X. For example, Tag is implemented as a type TagState +
 * a type Tag that inherits from Model<TagState>. (See design doc for an
 * explanation of this setup).
 *
 * Some methods in the template are compulsory, while others are
 * compulsory only if a particular component (lower/upper bounds and 
 * belief-update) included in the package is used. Such methods are marked
 * with the component(s) that require them, and are implemented with a
 * reasonable default, or an assert(false) which guarantees that if they are
 * required but not overridden in the subclass, the program will halt.
 *
 * Note: Some modules provided in the package have certain requirements on 
 * the user-defined state type. For example, UpperBoundStochastic requires 
 * the state type to have an implicit conversion to and from int because 
 * it uses state-indexed tables, and ExactBeliefUpdate requires the state 
 * type to be hashable in order to be able to work with transition matrices.
 * If your problem does not use these pre-defined modules, you must implement
 * dummy operations (e.g. see Pocman) to compile cleanly.
 * 
 * If the pre-defined lower/upper bound components are not used, you must 
 * implement the appropriate interface yourself. See RockSample for an example.
 */
template<typename T>
class Model {
 public:
  // The ctor is not part of the interface, so it can have any signature.
  // Typically it will accept problem-specific parameters like size and
  // number of elements, and a random number seed used to generate
  // the true starting state.
  
  virtual ~Model() { }

  // Modifies a state by taking an action on it, using a random number to
  // determine the resulting state, observation and reward.
  virtual void Step(T& state, double randomNum, int action, double& reward,
                    uint64_t& obs) const = 0;

  // Probability of receiving observation z given an action and the resulting
  // state. Used in belief updates.
  virtual double ObsProb(uint64_t z, const T& state, int action) const = 0;

  // True starting state of the world.
  virtual T GetStartState() const = 0;

  // Starting belief of the agent, represented as a mapping from state
  // to probability.
  virtual vector<pair<T, double>> InitialBelief() const = 0;

  // |A|
  virtual int NumActions() const = 0;

  // True if the state is a terminal state
  virtual bool IsTerminal(const T& s) const = 0;

  // A unique observation that must be emitted in and only in a terminal state.
  virtual uint64_t TerminalObs() const = 0;

  // Textual display
  virtual void PrintState(const T& state, ostream& out = cout) const = 0;
  virtual void PrintObs(uint64_t obs, ostream& out = cout) const = 0;

  // Methods to create and destroy particles. Override to manage memory
  // manually.
  virtual Particle<T>* Copy(const Particle<T>* particle) const {
    return new Particle<T>(*particle);
  }
  virtual Particle<T>* Allocate() const {
    return new Particle<T>();
  }
  virtual void Free(Particle<T>* particle) const {
    delete particle;
  }

  // The following methods don't all need to be implemented. Each method is
  // marked with the name of the component(s) that require it. Default
  // implementations are provided for some of them.

  // Required by: PolicyLowerBound
  // Lower bound at the fringe nodes (max depth) of the search tree.
  // Take care to ensure that this is a valid lower bound. For example, Tag
  // has negative rewards, so simply returning 0 for a set of particles that
  // are not all in the terminal state is incorrect.
  virtual double FringeLowerBound(const vector<Particle<T>*>& particles) const {
    assert(false);
    return 0;
  }

  // Required by: RandomPolicyLowerBound
  // Returns a preferred action for the given history. A random state in the
  // particle set of the history is also passed, only the observable part of
  // which may be used to to compute the action.
  // The action must be fixed for a given history.Return -1 if no preferred 
  // action exists.
  virtual int DefaultActionPreferred(const History& history, const T& state) 
      const {
    assert(false);
    return -1;
  }

  // Required by: RandomPolicyLowerBound
  // Returns a legal action for the given history. A random state in the
  // particle set of the history is also passed, only the observable part of
  // which may be used to to compute the action.
  // The action must be fixed for a given history.
  virtual int DefaultActionLegal(const History& history, const T& state) const {
    srand(history.Hash());
    return rand() % NumActions();
  }

  // Required by: ModePolicyLowerBound
  // Returns the best action for a given state.
  virtual int DefaultActionForState(const T& state) const {
    assert(false);
    return 0;
  };

  // Required by: UpperBoundStochastic
  // An overloaded version of Step() tha does not set an observation. In some 
  // cases this can give a significant speedup of the upper bound 
  // precomputation (e.g. see LaserTag). If no gains are to be made, simply 
  // call the previous version with a dummy observation (which is discarded).
  virtual void Step(T& state, double random_num, int action, double& reward) 
      const {
    uint64_t obs;
    Step(state, random_num, action, reward, obs);
  }

  // Required by: UpperBoundStochastic and UpperBoundNonStochastic
  // The upper bound at the fringe nodes (max depth) of the search tree.
  virtual double FringeUpperBound(const T& state) const {
    assert(false);
    return 0;
  }

  // Required by: UpperBoundStochastic and UpperBoundNonStochastic
  // |S|
  virtual int NumStates() const {
    assert(false);
    return 0;
  }

  // Required by: ParticleFilterUpdate
  // Returns a random state used in bootstrapping the particle filter when it
  // becomes empty. In the particle filter this method is called repeatedly, 
  // and the return value is checked for consistency with the observation @obs
  // until a sufficient number of consistent particles is obtained. Therefore,
  // although not necessary, returning states that are consistent with @obs 
  // will ensure that the filter refills quickly, especially in problems with 
  // large state spaces (e.g see Pocman).
  virtual T RandomState(unsigned& seed, uint64_t obs) const {
    assert(false);
    return T();
  }

  // Required by: ExactBeliefUpdate
  // Returns the transition matrix for the model, which is a mapping from
  // [state][action] to {resulting state, P(resulting state | state, action)}
  virtual vector<vector<UMAP<T, double>>> TransitionMatrix() const {
    assert(false);
    return {{{ }}};
  }
};

#endif
