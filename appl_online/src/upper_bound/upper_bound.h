#ifndef UPPER_BOUND_H
#define UPPER_BOUND_H

#include "globals.h"
#include "history.h"
#include "model.h"
#include "particle.h"
#include "random_streams.h"

// This is the interface implemented by all upper bound strategies.
template<typename T>
class IUpperBound {
 public:
  IUpperBound(const RandomStreams& streams) : streams_(streams) {}

  virtual ~IUpperBound() {};

  // Params: the history of the node, the set of particles at the node, the 
  // position of the particles in the random number stream (equal to the
  // depth in the tree), and the model.
  // Output: Upper bound
  virtual double UpperBound(History& history,
                            const vector<Particle<T>*>& particles,
                            int stream_position,
                            const Model<T>& model) const = 0;
 protected:
  const RandomStreams& streams_;
};

#endif
