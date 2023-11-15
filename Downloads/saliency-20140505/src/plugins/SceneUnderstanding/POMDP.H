/*!@file SceneUnderstanding/POMDP.H  partially observable Markov decision processes */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/POMDP.H $
// $Id: POMDP.H 13551 2010-06-10 21:56:32Z itti $
//

#ifndef POMDP_H_DEFINED
#define POMDP_H_DEFINED

#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H" // for uint
#include "Image/ImageSet.H" // for uint
#include "Image/Pixels.H" // for uint
#include <vector>
#include <string>
#include <map>

class POMDP
{
public:

  struct StateAction
  {
    int state;
    int action;
  };

  struct State
  {
    int x;
    int y;
    int stateID;
    float prob;

    State(int in_x, int in_y) :
      x(in_x), y(in_y)
    {
    }
  };

  enum OBJS {BACKGROUND, GOAL, AGENT, WALL, HOLE};
  enum ACTIONS {NORTH, SOUTH, EAST, WEST};
  POMDP();

  //! Destructor
  ~POMDP();

  //! which action to perform
  int getAction();

  //! make an observation, return if we have anything new
  bool makeObservation(const Image<PixRGB<byte> > &img);

  void init();


  //Observations
  Image<float> locateObject(const Image<float>& src, Image<float>& filter);
  Point2D<int> findObject(const Image<PixRGB<byte> > &img, const char* filename);
  std::vector<Point2D<int> > findMultipleObjects(const Image<PixRGB<byte> > &img, const char* filename);

  Point2D<int> getAgentState();
  Point2D<int> getGoalState();

  Image<byte> getStateSpace();
  void showTransitions();
  float getTransProb(int state, int action, int newState);  // p(s'|s,u)

  int doAction(const int state, const int act);
  float getReward(int state);
  void valueIteration();
  ACTIONS getAction(int state);

  //Get the proposed action
  int getPropAction();

  void doPolicy(const Point2D<int> &startPos);

  Image<float> getPerception(const uint obj);
  Image<float> makePrediction(const ACTIONS action);
  float updatePerception(const Image<PixRGB<byte> > &img);
  void updateStateTransitions(const ACTIONS action);
  bool goalReached();
  bool isExploring() { return itsExploring;}

  float bayesFilter(const int action, const Image<PixRGB<byte> > &img);
  float particleFilter(const int action, const Image<PixRGB<byte> > &img);

  float getEntropy(Image<float> &belif);

  //! The model of the object
  float getObjProb(const Image<PixRGB<byte> > &img,
      const State state, const int objID);




private:
    Dims itsRetinaSize;
    Point2D<int> itsAgentState;
    Point2D<int> itsGoalState;

    float itsLastDistance;
    int itsLastAction;
    Image<byte> itsStateSpace;
    ImageSet<float> itsCurrentPercep;
    Image<float> itsPreviousPercep;
    Image<float> itsPrediction;
    std::vector<std::vector<std::vector<float> > > itsTransitions;
    Image<float> itsUtility;
    std::map<StateAction, int> itsStateTrans;
    bool itsExploring;
    std::vector<Image<PixRGB<byte> > > itsObjects;

    std::vector<State> itsParticleStateSpace;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
