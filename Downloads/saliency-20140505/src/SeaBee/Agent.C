/*!@file SeaBee/Agent.C  base class for agents (has run function)       */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/Agent.C $
// $Id: Agent.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <unistd.h>
#include "Util/log.H"

#include "Agent.H"

// ######################################################################
//Constructors
Agent::Agent(std::string name)
{
  itsName = name;
  itsLastAction = "";
  pthread_mutex_init(&itsStateChangedLock, NULL);
  stateChange = 0;
}

// ######################################################################
Agent::~Agent() { }

// ######################################################################
//Prints an action being performed by an agent
void Agent::Do(std::string msg)
{
  if(itsLastAction != msg)
    {
      std::cout<<">>>"<<itsName<<"::"<<msg<<"<<<"<<std::endl;
      itsLastAction = msg;
    }
}

// ######################################################################
//Dummy scheduler. Descendent agents implement something meaningful here
bool Agent::pickAndExecuteAnAction()
{
  return false;
}

// ######################################################################
//Indicates that something has occured to change the state of the agent
void Agent::stateChanged()
{
  pthread_mutex_lock(&itsStateChangedLock);
  stateChange++;
  pthread_mutex_unlock(&itsStateChangedLock);
}

// ######################################################################
//Used to put agent to sleep until stateChanged is called
void Agent::acquire()
{
  pthread_mutex_lock(&itsStateChangedLock);
  stateChange--;
  if(stateChange < 0)
  {
    stateChange = 0;
  }
  pthread_mutex_unlock(&itsStateChangedLock);
}

// ######################################################################
// starts running the agent
void Agent::run()
{
  bool goOn = true;
  while(goOn)
  {
    // while the agent has stuff to do,
    // call the scheduler
    while(pickAndExecuteAnAction());

    // puts the agent to sleep
    // until another agent wakes it up with a message
    acquire();
    while(isAsleep()) { usleep(10000); }
  }
}

// ######################################################################
//Returns whether or not the agent is asleep as indicated by stateChange
bool Agent::isAsleep()
{

  pthread_mutex_lock(&itsStateChangedLock);
  bool sleep = (stateChange == 0);
  pthread_mutex_unlock(&itsStateChangedLock);

  return sleep;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */




