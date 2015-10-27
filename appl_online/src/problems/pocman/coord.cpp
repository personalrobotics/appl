#include "coord.h"

const Coord Coord::Null(-1, -1);
const Coord Coord::North(0, 1);
const Coord Coord::East(1, 0);
const Coord Coord::South(0, -1);
const Coord Coord::West(-1, 0);
const Coord Coord::NorthEast(1, 1);
const Coord Coord::SouthEast(1, -1);
const Coord Coord::SouthWest(-1, -1);
const Coord Coord::NorthWest(-1, 1);

const Coord Coord::Compass[8] = 
{ 
    North, 
    East, 
    South, 
    West, 
    NorthEast,
    SouthEast,
    SouthWest,
    NorthWest
};

const char* Coord::CompassString[8] = 
{ 
    "N", 
    "E",
    "S",
    "W",
    "NE",
    "SE",
    "SW",
    "NW"
};

void Coord::UnitTest()
{
    assert(Coord(3, 3) + Coord(2, 2) == Coord(5, 5));
    Coord coord(5, 2);
    coord += Coord(2, 5);
    assert(coord == Coord(7, 7));
    assert(Coord(2, 2) + North == Coord(2, 3));
    assert(Coord(2, 2) + East == Coord(3, 2));
    assert(Coord(2, 2) + South == Coord(2, 1));
    assert(Coord(2, 2) + West == Coord(1, 2));
    assert(Compass[E_NORTH] == North);
    assert(Compass[E_EAST] == East);
    assert(Compass[E_WEST] == West);
    assert(Compass[E_SOUTH] == South);
    assert(Clockwise(E_NORTH) == E_EAST);
    assert(Clockwise(E_EAST) == E_SOUTH);
    assert(Clockwise(E_SOUTH) == E_WEST);
    assert(Clockwise(E_WEST) == E_NORTH);
    assert(Opposite(E_NORTH) == E_SOUTH);
    assert(Opposite(E_EAST) == E_WEST);
    assert(Opposite(E_SOUTH) == E_NORTH);
    assert(Opposite(E_WEST) == E_EAST);
    assert(Anticlockwise(E_NORTH) == E_WEST);
    assert(Anticlockwise(E_EAST) == E_NORTH);
    assert(Anticlockwise(E_SOUTH) == E_EAST);
    assert(Anticlockwise(E_WEST) == E_SOUTH);
    assert(ManhattanDistance(Coord(3, 2), Coord(-4, -7)) == 16);
    assert(DirectionalDistance(Coord(3, 2), Coord(-4, -7), E_NORTH) == -9);
    assert(DirectionalDistance(Coord(3, 2), Coord(-4, -7), E_EAST) == -7);
    assert(DirectionalDistance(Coord(3, 2), Coord(-4, -7), E_SOUTH) == 9);
    assert(DirectionalDistance(Coord(3, 2), Coord(-4, -7), E_WEST) == 7);
}
