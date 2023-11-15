#ifndef CRAPPY_COMPASS_H_DEFINED
#define CRAPPY_COMPASS_H_DEFINED

#include <vector>
#include "Devices/BeeStemTiny.H"


//! The crappyCompass class can map crappy compass values to real headings as long as the crappy input is consistent.
//  It does this by running a calibration routined which reads in as many values as possible and stores them in a
//  vector. Then, when a user asks the crappyCompass class for a real heading, it passes it a crappy value, and
//the class will map it to a point in its crappy vector, returning the index.
class crappyCompass
{
public:
  crappyCompass(nub::soft_ref<BeeStemTiny> &b) : itsBeeStem(b){

  }

  void calibrate();

private:
  nub::soft_ref<BeeStemTiny> itsBeeStem;
  std::vector raw_readings;
}


#endif
