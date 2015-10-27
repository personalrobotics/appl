#ifndef PARTICLE_H
#define PARTICLE_H

#include "memorypool.h"

template<typename T>
class Particle : public MemoryObject {
 public:
  T state;
  int id;
  double wt;
};

#endif
