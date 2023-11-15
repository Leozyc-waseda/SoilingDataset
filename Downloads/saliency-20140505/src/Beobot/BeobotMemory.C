/*!@file Beobot/BeobotMemory.C sensory/motor memory for the Beobots */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotMemory.C $
// $Id: BeobotMemory.C 4663 2005-06-23 17:47:28Z rjpeters $
//

#include "Beobot/BeobotMemory.H"

#include "Util/Assert.H"

// #########################################################################
BeobotMemory::BeobotMemory( void )
{
  idx = 0;
  state = HAVE_ACTION;
  // just to initialize, this way we can only call presentSensation(...)

  for(int t=0;t<memoryLength;t++)
    valid[t]=false;
}

// #########################################################################
void BeobotMemory::presentSensation( BeobotSensation &sensa )
{
  // check that we are begining a new cycle
  ASSERT( state == HAVE_ACTION );

  sensation[idx] = sensa;
  valid[idx] = true;

  state =  HAVE_SENSATION;

  // we will idx++ at the end !
}

// #########################################################################
void BeobotMemory::presentActionRecommendation( BeobotAction &act )
{
  // the actionRecommendation is based on previous sensations
  ASSERT( state == HAVE_SENSATION );

  actionRecommendation[ idx ] = act;

  state = HAVE_RECOM_ACTION;
}

// #########################################################################
bool BeobotMemory::passedSensation( int howOld, BeobotSensation & sensa )
{
  ASSERT( howOld < memoryLength );
  int i = idx - howOld; if (i < 0) i += memoryLength;
  // note: no -1 because we have not incremented idx

  if(!valid[i])
    return false;

  sensa = sensation[i];
  return true;
}

// #########################################################################
void BeobotMemory::generatePresentAction( BeobotAction & currentAction )
{
  ASSERT( state == HAVE_RECOM_ACTION );

  float turn;
  float speed;
  int gear;

  gear=1; // why not ?

  // computes the new action as
  turn=0;
  speed=0;
  int nbValidMemories=0;
  for(int t=0;t<memoryLength;t++)
    {
      if( valid[t] )
        // if( ! confused )
        /*
          try to think of something clever here
          if sensations are very different then maybe we need more
          time.
        */
        {
          turn+=actionRecommendation[t].getTurn();
          speed+=actionRecommendation[t].getSpeed();
          nbValidMemories++;
        }
    }
  currentAction.setTurn( turn/nbValidMemories );
  currentAction.setSpeed( speed/nbValidMemories );
  currentAction.setGear( gear );

  // keep track of our decision
  action[idx]=currentAction;

  // consider oldest cell as next free cell
  idx++;
  if (idx >= memoryLength) idx = 0;

  state = HAVE_ACTION;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
