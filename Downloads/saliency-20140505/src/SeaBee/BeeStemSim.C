/*!@file Devices/BeeStemSim.C Simple interface to beestem */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/BeeStemSim.C $
// $Id: BeeStemSim.C 10794 2009-02-08 06:21:09Z itti $
//

#include "SeaBee/BeeStemSim.H"

#include "Component/OptionManager.H"
#include "Devices/Serial.H"
#include "Image/MatrixOps.H"
#include <string>

#define BS_CMD_DELAY 5000000

namespace
{
  class SubSimLoop : public JobWithSemaphore
  {
    public:
      SubSimLoop(BeeStemSim* beeStemSim)
        :
          itsBeeStemSim(beeStemSim),
          itsPriority(1),
          itsJobType("controllerLoop")
    {}

      virtual ~SubSimLoop() {}

      virtual void run()
      {
        ASSERT(itsBeeStemSim);
        while(1)
        {
          itsBeeStemSim->simLoop();
          usleep(1000);
        }
      }

      virtual const char* jobType() const
      { return itsJobType.c_str(); }

      virtual int priority() const
      { return itsPriority; }

    private:
      BeeStemSim* itsBeeStemSim;
      const int itsPriority;
      const std::string itsJobType;
  };
}



// ######################################################################
BeeStemSim::BeeStemSim(OptionManager& mgr, const std::string& descrName,
         const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsSubSim(new SubSim(mgr))
{
  // attach our port as a subcomponent:
  addSubComponent(itsSubSim);

  for (int i=0; i<5; i++)
    itsLastMotorCmd[i] = 0;

  initRandomNumbers();
}

// ######################################################################
BeeStemSim::~BeeStemSim()
{
}

void BeeStemSim::start2()
{
  //setup pid loop thread
  itsThreadServer.reset(new WorkThreadServer("SubSim",1)); //start a single worker thread
  itsThreadServer->setFlushBeforeStopping(false);
  rutz::shared_ptr<SubSimLoop> j(new SubSimLoop(this));
  itsThreadServer->enqueueJob(j);
}

void BeeStemSim::simLoop()
{
  itsSubSim->simLoop();
  //itsForwardImg = itsSubSim->getFrame(1);
  //itsDownImg = itsSubSim->getFrame(2);

}

// ######################################################################
bool BeeStemSim::setThrusters(int &m1, int &m2, int &m3,
                int &m4, int &m5)
{
  // Command Buffer:
  // [0] is start character
  // [1..5] is m1 m5

  if (m1 > MOTOR_MAX) m1 = MOTOR_MAX; if (m1 < -MOTOR_MAX) m1 = -MOTOR_MAX;
  if (m2 > MOTOR_MAX) m2 = MOTOR_MAX; if (m2 < -MOTOR_MAX) m2 = -MOTOR_MAX;
  if (m3 > MOTOR_MAX) m3 = MOTOR_MAX; if (m3 < -MOTOR_MAX) m3 = -MOTOR_MAX;
  if (m4 > MOTOR_MAX) m4 = MOTOR_MAX; if (m4 < -MOTOR_MAX) m4 = -MOTOR_MAX;
  if (m5 > MOTOR_MAX) m5 = MOTOR_MAX; if (m5 < -MOTOR_MAX) m5 = -MOTOR_MAX;



  if (abs(itsLastMotorCmd[0] - m1) > 60 && itsLastMotorCmd[0]*m1 < 0) m1 = 0;
  if (abs(itsLastMotorCmd[1] - m2) > 60 && itsLastMotorCmd[1]*m2 < 0) m2 = 0;
  if (abs(itsLastMotorCmd[2] - m3) > 60 && itsLastMotorCmd[2]*m3 < 0) m3 = 0;
  if (abs(itsLastMotorCmd[3] - m4) > 60 && itsLastMotorCmd[3]*m4 < 0) m4 = 0;
  if (abs(itsLastMotorCmd[4] - m5) > 60 && itsLastMotorCmd[4]*m5 < 0) m5 = 0;


  itsLastMotorCmd[0] = m1;
  itsLastMotorCmd[1] = m2;
  itsLastMotorCmd[2] = m3;
  itsLastMotorCmd[3] = m4;
  itsLastMotorCmd[4] = m5;

  float pan = ((float(m2) - float(m4))/2) / 100.0;
  float tilt = 0; //((((float(m1) + float(m5))/2) - float(m3))/2.0)/100.0;
  float forward = -1*((float(m2) + float(m4))/2) / 100.0;
  float up = -1*((float(m1) + float(m3) + float(m5))/3.0)/100.0;


  itsSubSim->setTrusters(pan, tilt, forward, up);


  return true;

}

// ######################################################################
bool BeeStemSim::getSensors(int &heading, int &pitch, int &roll, int &ext_pressure, int &int_pressure)
{
  float simxPos;
  float simyPos;
  float simdepth;
  float simroll;
  float simpitch;
  float simyaw;

  int heading_noise = 0;
  int pitch_noise = randomUpToIncluding(6) - 3;
  int roll_noise = randomUpToIncluding(6) - 3;
  int ext_p_noise = randomUpToIncluding(6) - 3;
  int int_p_noise = randomUpToIncluding(6) - 3;

  itsSubSim->getSensors(simxPos, simyPos, simdepth,
      simroll, simpitch, simyaw);

  heading = (int)(simyaw*180/M_PI) + heading_noise;
  pitch = (int)(simpitch*180/M_PI) + pitch_noise;
  roll = (int)(simroll*180/M_PI) + roll_noise;

  ext_pressure = (int)(simdepth*100) + ext_p_noise;

  int_pressure = 90 + int_p_noise;


  return true;
}

bool BeeStemSim::setHeartBeat()
{
  return true;

}

Image<PixRGB<byte> > BeeStemSim::getImage(int camera)
{
//   switch(camera)
//     {
//     case 1: return flipVertic(itsForwardImg);
//     case 2: return flipVertic(itsDownImg);
//     }

  return flipVertic(itsSubSim->getFrame(camera));

//  return Image<PixRGB<byte> > ();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
