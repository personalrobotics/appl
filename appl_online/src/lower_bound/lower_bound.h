#ifndef LOWERBOUND_H
#define LOWERBOUND_H

#include "globals.h"
#include "history.h"
#include "model.h"
#include "particle.h"
#include "random_streams.h"

// The interface implemented by lower bound strategies.
template<typename T>
class ILowerBound {
 public:
  ILowerBound(const RandomStreams& streams) : streams_(streams) {}

  virtual ~ILowerBound() {}
  
  // Params: the history of the node, the set of particles at the node, the
  // position of the particles in the random number stream, (equal to the
  // depth in the search tree), and the model.
  // Output: The lower bound and the first-step action that achieves the bound.
  virtual pair<double, int> LowerBound(History& history,
                                       const vector<Particle<T>*>& particles,
                                       int stream_position,
                                       const Model<T>& model) const = 0;

 protected:
  const RandomStreams& streams_;
};

#endif
