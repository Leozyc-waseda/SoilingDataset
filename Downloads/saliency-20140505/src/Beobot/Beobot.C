/*!@file Beobot/Beobot.C main Beobot class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Beobot.C $
// $Id: Beobot.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Beobot/Beobot.H"

// ######################################################################
Beobot::Beobot(const char *slaves, const int imgw, const int imgh,
               const int lev_min, const int lev_max,
               const int delta_min, const int delta_max,
               const int smlev, const int nborient, const MaxNormType normtype,
               const int jlev, const int jdepth, const int nbneig,
               const in_addr_t ip, const short int port)
{
  if (slaves)
    {
      // parallel version:
      LINFO("Initializing Beowulf as master");

      LFATAL("THIS CODE IS TEMPORARILY BROKEN");

      visualCortex.init(imgw, imgh, lev_min, lev_max, delta_min, delta_max,
                        smlev, nborient, normtype, jlev, jdepth, nbneig, beo);
    }
  else
    // single-CPU version:
    visualCortex.init(imgw, imgh, lev_min, lev_max, delta_min, delta_max,
                      smlev, nborient, normtype, jlev, jdepth, nbneig,
                      nub::soft_ref<Beowulf>());
}

// ######################################################################
void Beobot::newVisualInput(Image< PixRGB<byte> >& scene)
{ visualCortex.newVisualInput(scene); }

// ######################################################################
Image< PixRGB<byte> >* Beobot::getRetinaPtr()
{ return visualCortex.getScenePtr(); }

// ######################################################################
void Beobot::lowLevel(const int frame)
{ visualCortex.process(frame); }

// ######################################################################
void Beobot::lowLevelStart(const int frame)
{ visualCortex.processStart(frame); }

// ######################################################################
void Beobot::lowLevelEnd(const int frame)
{ visualCortex.processEnd(frame); }

// ######################################################################
void Beobot::getWinner(Point2D<int>& win) const
{ visualCortex.getWinner(win); }

//###########################################################################
void Beobot::intermediateLevel(bool initMasses)
{
  visualCortex.initSprings(initMasses);

  const float dt = 0.1;

  int nbIter = 3; if (initMasses) nbIter=15;

  for (int t = 0; t < nbIter; t ++) visualCortex.iterateSprings(dt);

  Image< PixRGB<byte> > clusteredImage;
  Point2D<int> supposedTrackCentroid;
  Point2D<int> previousTrackCentroid;

  BeobotSensation sensa;

  // carefull : passedSensation( 0, sensa ) is the previous
  // sensation as we will update the memory only at the end
  // of this function...

  // the number of valid memories
  int nbMemories = 0;

  while (memory.passedSensation(nbMemories, sensa))
    {
      Point2D<int> pt;
      sensa.getCentroid(pt);

      previousTrackCentroid += pt;
      nbMemories ++;

      if (nbMemories >= BeobotMemory::memoryLength) break;
    }

  if (nbMemories > 0)
    // take the mean of previous centroids
    previousTrackCentroid /= nbMemories;
  else
    {
      // else, just pretend it was where it should be ;)

      visualCortex.getInputSize(previousTrackCentroid);

      previousTrackCentroid.i /= 2;
      // previousTrackCentroid.j *= 7/8
      previousTrackCentroid.j *= 7; previousTrackCentroid.j /= 8;
    }

  visualCortex.getClusteredImage(clusteredImage, supposedTrackCentroid,
                                 previousTrackCentroid);

  BeobotSensation sensation(clusteredImage, supposedTrackCentroid);

  memory.presentSensation(sensation);
}

//###########################################################################
void Beobot::DEBUGgetClustered(Image< PixRGB<byte> >& im)
{
  BeobotSensation sensa;
  memory.passedSensation(0, sensa);

  sensa.getQuickLayout(im);
}

//###########################################################################
void Beobot::highLevel( void )
{
  BeobotSensation s;

  memory.passedSensation(0, s);
  Image< PixRGB<byte> > ql;

  s.getQuickLayout(ql);

  // size of the image
  int w = ql.getWidth();
  int h = ql.getWidth();

  // 'point' is at the center of the bottom of the image
  int pointX = w / 2;
  int pointY = h * 7 / 8;

  Point2D<int> centroid;
  s.getCentroid(centroid);

  // create a new action
  BeobotAction action;

  // SPEED
  if (centroid.j > pointY)
    // not much visibility, go slow
    action.setSpeed(0.1);
  else if( centroid.j < h/2 )
    action.setSpeed(1.0);
  else
    // do something linear in between
    // note: centroid.j should be > h/2
    action.setSpeed( 1.0*( -(float)centroid.j+(float)h )/((float)h/2.0)  );

  // TURN
  // linear is beautiful
  action.setTurn( ((float)centroid.i-(float)pointX)/(float)pointX  );

  // GEAR
  action.setGear(1);
  // we'll think about this one later...

  memory.presentActionRecommendation( action );
}

//###########################################################################
void Beobot::decision( void )
{ // all the work is done by memory
  memory.generatePresentAction(currentAction);
}

//###########################################################################
void Beobot::action( void )
{
  //LINFO("Beobot::act()");

// all the work is done by effectors
  effectors->performAction(currentAction);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
