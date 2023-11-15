/*!@file SeaBee/Agent.H
 base class for agents. Has run function for thread running             */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/Agent.H $
// $Id: Agent.H 9797 2008-06-07 22:09:04Z beobot $
//
//////////////////////////////////////////////////////////////////////////

#ifndef __AGENT_H__
#define __AGENT_H__

#include <pthread.h>
#include <string>

#include "Globals.H"

class Agent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  Agent(std::string name);
  virtual ~Agent();

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! get the name of the agent
  inline std::string getName();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  virtual bool pickAndExecuteAnAction();
  bool isAsleep();
  void acquire();
  void run();

  void Do(std::string msg);

  //@}


protected:

  //! updates the agent's stateChange
  void stateChanged();
  pthread_mutex_t itsStateChangedLock;

  //! the name of the agent
  std::string itsName;

  //! indicates that the agent's state has changed
  volatile int stateChange;

  std::string itsLastAction;

};

// ######################################################################
// Implementation for Agent inline functions
// ######################################################################
inline std::string Agent::getName()
{
  return itsName;
}
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */



