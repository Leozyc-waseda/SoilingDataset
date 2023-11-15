/*!@file Neuro/PrefrontalCortexService.H */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/PrefrontalCortexService.H $
// $Id: PrefrontalCortexService.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef NEOVISIONII_PREFORNTALCORTEXSERVICE_H_DEFINED
#define NEOVISIONII_PREFORNTALCORTEXSERVICE_H_DEFINED

#include <Ice/Ice.h>
#include <Ice/Service.h>
#include <IceUtil/Thread.h>
#include <IceStorm/IceStorm.h>
#include "Ice/BrainObjects.ice.H"
#include "Ice/SimEvents.ice.H"
#include "Ice/ImageIce.ice.H"
#include "Ice/IceImageUtils.H"
#include "Component/ModelManager.H"
#include "Component/ModelComponent.H"
#include "Component/ModelOptionDef.H"
#include "Image/Image.H"
#include "Image/Point3D.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"

class PrefrontalCortexI : public ModelComponent, public BrainObjects::PrefrontalCortex
{
public:

  enum STATES {START_OF_GAME_MSG, WAIT_FOR_GAME_START,
               INIT_PTZ, INIT_PTZ_DONE,
               SET_TRACKING, TRACKING, SET_SEG_LOC, GET_SEG_IMG,
               GET_OBJ_INFO, GET_OBJ_STATE, ZOOM_OUT, ZOOM_OUT_DONE,
               GAMEPLAY_PLAYERMOVE, GAMEPLAY_PLAYERMOVE_DONE};



  struct TopicInfo
  {
    TopicInfo(const char* _name, IceStorm::TopicPrx& prx) :
      name(_name), topicPrx(prx)
    {}
    std::string name;
    IceStorm::TopicPrx topicPrx;
  };

  struct StateInfo
  {
    StateInfo(int _id, const char* _name, int _nextState) :
      id(_id), name(_name), nextState(_nextState)
    {}

    int id;
    std::string name;
    int nextState;
  };

  PrefrontalCortexI(OptionManager& mgr,
        const std::string& descrName = "PrefrontalCortexService",
        const std::string& tagName = "PrefrontalCortexService");

  ~PrefrontalCortexI();

  virtual void evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&);

  void initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx); //Init the SimEvents by attaching to a topic and getting a messagePrx
  void unsubscribeSimEvents();

  //State methods
  void assignStates(); //assign states
  void setCurrentState(int s); //set the current state
  int processState(int state,  const SimEvents::EventMessagePtr& eMsg);

  void setSalientTrackLoc(const SimEvents::SaliencyMapMessagePtr& msg);
  void setTrackLoc(Point2D<int> loc);
  void setTrackSegLoc(const SimEvents::VisualTrackerMessagePtr& msg);
  void setPTZoomIn();

  void init();

private:
  SimEvents::EventsPrx itsEventsPub; //the Event publisher
  std::vector<TopicInfo> itsTopicsSubscriptions; //out subscription to the various topics
  Ice::ObjectPrx itsObjectPrx; //Our proxy
  std::vector<StateInfo> itsStates; //Our state info vectors
  int itsCurrentState;      //our current state
  int itsLastState;      //The last state we where in
  Image<PixRGB<byte> > itsCurrentRetinaImg; //The last retina image
  int itsTilt;      //current tilt
  int itsPan;       //current pan;
  int itsCount;     //Count
  bool getUserInput;
  int itsArmXOffset;
  int itsArmYOffset;
  int itsArmZOffset;

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // NEURO_ENVVISUALCORTEX_H_DEFINED
