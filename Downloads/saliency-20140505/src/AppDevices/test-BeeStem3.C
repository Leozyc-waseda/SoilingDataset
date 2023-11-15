/*!@file AppDevices/test-BeoChip.C test suite for Rand Voorhies' BeeSTEM */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-BeeStem3.C $
// $Id: test-BeeStem3.C 12977 2010-03-07 10:58:57Z beobot $
//

#include "Component/ModelManager.H"
#include "Devices/BeeStem3.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"

#include <cstdlib>
#include <iostream>

int main(const int argc, const char* argv[])
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("BeeStem3 Test Program");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeeStem3> b(new BeeStem3(manager,"BeeStem3", "BeeStem3", "/dev/ttyUSB0"));
  manager.addSubComponent(b);

  if (manager.parseCommandLine(argc, argv, "No Options Yet", 0,0) == false)
                                   return(1);


  // let's get all our ModelComponent instances started:
  manager.start();

  int accelX,accelY,accelZ;
  int compassHeading, compassPitch, compassRoll;
  int internalPressure, externalPressure;
  int desiredHeading, desiredDepth, desiredSpeed;
  int headingK, headingP, headingD, headingI, headingOutput;
  int depthK, depthP, depthD, depthI, depthOutput;
  char killSwitch;

  char c = 0;

  /*  int dheading = 0;
  int ddepth = 0;
  int dspeed = 0;
  //  int dmarker = 0;

  int pidMode = 0; 
  int dk = 0;
  int dp = 0;
  int di = 0;
  int dd = 0;*/

  while(c != 'q')
    {
      std::cout << "Command (r = read, s = start compass calibration, e = end compass calibration, q = quit): ";
      std::cin >> c;
      std:: cout << "\n";
      
      if(c == 'r')
        {
          while(b->getSensors(accelX,accelY,accelZ,
                  compassHeading, compassPitch, compassRoll,
                  internalPressure, externalPressure,
                  desiredHeading, desiredDepth, desiredSpeed,
                  headingK, headingP, headingD, headingI, headingOutput,
                  depthK, depthP, depthD, depthI, depthOutput, killSwitch))
            {
              LINFO("===================================");
              LINFO("Accel X: %d, AccelY: %d, AccelZ: %d",accelX,accelY,accelZ);
              LINFO("Heading: %d, Pitch: %d, Roll: %d",compassHeading, compassPitch, compassRoll);
              LINFO("Int Pressure: %d, Ext Pressure: %d",internalPressure,externalPressure);
              LINFO("Desired Heading: %d, Desired Depth: %d, Desired Speed: %d",desiredHeading, desiredDepth, desiredSpeed);
              LINFO("Heading K=%d, P=%d, I=%d, D=%d, Output=%d",headingK, headingP, headingI, headingD, headingOutput);
              LINFO("Depth K=%d, P=%d, I=%d, D=%d, Output=%d",depthK, depthP, depthI, depthD, depthOutput);
              LINFO("Kill Switch: %d",killSwitch);
              LINFO("===================================");
              sleep(0.5);
            }
          
        }
      else if(c == 's')
        {
          LINFO("Starting compass calibration");
          b->startCompassCalibration();
        }
      else if(c == 'e')
        {
          LINFO("Ending compass calibration");
          b->endCompassCalibration();                    
        }
      /*else if(c == 's')
        {
          std::cout << "Desired Heading: ";
          std::cin >> dheading;
          std::cout << "\n";
          std::cout << "Desired Depth: ";
          std::cin >> ddepth;
          std::cout << "\n";
          std::cout << "Desired Speed: ";
          std::cin >> dspeed;
          std::cout << "\n";
          
          // NOTE: dmarker is ignored for now

          bool succ = b->setDesiredValues(dheading,ddepth,dspeed,dmarker);

          if(succ)
            LINFO("Desired values updated successfully!");
          else
          LINFO("Error setting desired values");

        }
      else if(c == 'i')
        {
          std::cout << "PID Mode (0 = Depth, 1 = Heading): "; 
          std::cin >> pidMode;
          std::cout <<"\n";
          
          std::cout << "K = "; 
          std::cin >> dk;
          std::cout <<"\n";
          
          std::cout << "P = "; 
          std::cin >> dp;
          std::cout <<"\n";
          
          std::cout << "I = "; 
          std::cin >> di;
          std::cout <<"\n";
          
          std::cout << "D = "; 
          std::cin >> dd;
          std::cout <<"\n";

          bool succ = b->setPID(pidMode,dk,dp,di,dd);

          if(succ)
            LINFO("PID values updated successfully!");
          else
            LINFO("Error setting PID values");
        }*/
}
  
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
