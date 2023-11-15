/*!@file AppDevices/test-gyro.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-gyro.C $
// $Id: test-gyro.C 6182 2006-01-31 18:41:41Z rjpeters $

#include "Devices/Gyro.H"
#include <iostream>
#include <unistd.h>

int main()
{
  Gyro gyr;
  int x,y;
  while(1)
    {
      gyr.getAngle(x,y);
      std::cout<<x<<" "<<y<<std::endl;
      usleep(30000);
    }

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
