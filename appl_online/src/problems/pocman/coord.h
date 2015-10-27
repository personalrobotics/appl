#ifndef COORD_H
#define COORD_H

#include <stdlib.h>
#include <assert.h>
#include <ostream>
#include <math.h>

struct Coord
{
  int X, Y;
  
  Coord() {}

  Coord(int x, int y) : X(x), Y(y) {}

  bool Valid() const {
    return X >= 0 && Y >= 0;
  }

  bool operator==(Coord rhs) const {
    return X == rhs.X && Y == rhs.Y;
  }

  bool operator<(const Coord& other) const {
    return X < other.X || (X == other.X && Y < other.Y);
  }
  
  bool operator!=(Coord rhs) const {
    return X != rhs.X || Y != rhs.Y;
  }

  void operator+=(Coord offset) {
    X += offset.X;
    Y += offset.Y;
  }

  Coord operator+(Coord rhs) const {
    return Coord(X + rhs.X, Y + rhs.Y);
  }
  
  Coord operator*(int mul) const
  {
    return Coord(X * mul, Y * mul);
  }

  enum {
    E_NORTH,
    E_EAST,
    E_SOUTH,
    E_WEST,
    E_NORTHEAST,
    E_SOUTHEAST,
    E_SOUTHWEST,
    E_NORTHWEST
  };
  
  static double EuclideanDistance(Coord lhs, Coord rhs);
  static int ManhattanDistance(Coord lhs, Coord rhs);
  static int DirectionalDistance(Coord lhs, Coord rhs, int direction);
  
  static const Coord Null;
  static const Coord North, East, South, West;
  static const Coord NorthEast, SouthEast, SouthWest, NorthWest;
  static const Coord Compass[8];
  static const char* CompassString[8];
  static int Clockwise(int dir) { return (dir + 1) % 4; }
  static int Opposite(int dir) { return (dir + 2) % 4; }
  static int Anticlockwise(int dir) { return (dir + 3) % 4; }

  static void UnitTest();
};

inline double Coord::EuclideanDistance(Coord lhs, Coord rhs) {
  return sqrt((lhs.X - rhs.X) * (lhs.X - rhs.X) + 
              (lhs.Y - rhs.Y) * (lhs.Y - rhs.Y));
}

inline int Coord::ManhattanDistance(Coord lhs, Coord rhs) {
  return abs(lhs.X - rhs.X) + abs(lhs.Y - rhs.Y);
}

inline int Coord::DirectionalDistance(Coord lhs, Coord rhs, int direction) {
  switch (direction) {
    case E_NORTH: return rhs.Y - lhs.Y;
    case E_EAST: return rhs.X - lhs.X;
    case E_SOUTH: return lhs.Y - rhs.Y;
    case E_WEST: return lhs.X - rhs.X;
    default: assert(false);
  }
}

inline std::ostream& operator<<(std::ostream& ostr, Coord& coord) {
  ostr << "(" << coord.X << ", " << coord.Y << ")";
  return ostr;
}

#endif // COORD_H
