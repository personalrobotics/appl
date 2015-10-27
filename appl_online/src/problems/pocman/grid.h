#ifndef GRID_H
#define GRID_H

#include "problems/pocman/coord.h"

template <class T>
class Grid {
 public:
  Grid() { }

  Grid(int xsize, int ysize) 
      : xsize_(xsize), 
        ysize_(ysize) {
    grid_.resize(xsize * ysize);
  }

  void Resize(int xsize, int ysize) {
    xsize_ = xsize;
    ysize_ = ysize;
    grid_.resize(xsize * ysize);
  }

  int GetXSize() const { return xsize_; }
  int GetYSize() const { return ysize_; }
  
  T& operator()(int index) {
    assert(index >= 0 && index < xsize_ * ysize_);
    return grid_[index];
  }

  const T& operator()(int index) const {
    assert(index >= 0 && index < xsize_ * ysize_);
    return grid_[index];
  }

  T& operator()(const Coord& coord) {
    assert(Inside(coord));
    return grid_[Index(coord)];
  }
  
  const T& operator()(const Coord& coord) const {
    assert(Inside(coord));
    return grid_[Index(coord)];
  }

  T& operator()(int x, int y) {
    assert(Inside(Coord(x, y)));
    return grid_[Index(x, y)];
  }
  
  const T& operator()(int x, int y) const {
    assert(Inside(Coord(x, y)));
    return grid_[Index(x, y)];
  }
  
  int Index(const Coord& coord) const {
    return xsize_ * coord.Y + coord.X;
  }

  int Index(int x, int y) const {
    assert(Inside(Coord(x, y)));
    return xsize_ * y + x;
  }
  
  bool Inside(const Coord& coord) const {
    return coord.X >= 0 && coord.Y >= 0 && 
           coord.X < xsize_ && coord.Y < ysize_;
  }

  int DistToEdge(const Coord& coord, int direction) {
    assert(Inside(coord));
    switch (direction) {
      case Coord::E_NORTH: return ysize_ - 1 - coord.Y;
      case Coord::E_EAST: return xsize_ - 1 - coord.X;
      case Coord::E_SOUTH: return coord.Y;
      case Coord::E_WEST: return coord.X;
      default: assert(false);
    }
  }

  void SetAllValues(const T& value) {
    for (int x = 0; x < xsize_; x++)
      for (int y = 0; y < ysize_; y++)
        grid_[Index(x, y)] = value;
  }
  
  void SetRow(int y, T* values) {
    for (int x = 0; x < xsize_; x++)
      grid_[Index(x, y)] = values[x];
  }

  void SetCol(int x, T* values) {
    for (int y = 0; y < ysize_; y++)
      grid_[Index(x, y)] = values[y];
  }    
  
  Coord IndexToCoord(int index) const {
    assert(index >= 0 && index < xsize_ * ysize_);
    return Coord(index % xsize_, index / xsize_);
  }
  
 private:
  int xsize_, ysize_;
  std::vector<T> grid_;
};

#endif // GRID_H
