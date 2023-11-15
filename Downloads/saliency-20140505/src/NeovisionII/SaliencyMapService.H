/*!@file Neuro/SaliencyMapService.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/SaliencyMapService.H $
// $Id: SaliencyMapService.H 12281 2009-12-17 09:00:36Z itti $
//

#ifndef NEOVISIONII_SaliencyMapSERVICE_H_DEFINED
#define NEOVISIONII_SaliencyMapSERVICE_H_DEFINED

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
#include "Neuro/EnvSaliencyMap.H"
#include "Image/Image.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"

class SaliencyMapI : public ModelComponent, public BrainObjects::SaliencyMap
{
public:
  struct TopicInfo
  {
    TopicInfo(const char* _name, IceStorm::TopicPrx& prx) :
      name(_name), topicPrx(prx)
    {}
    std::string name;
    IceStorm::TopicPrx topicPrx;
  };

  SaliencyMapI(OptionManager& mgr,
        const std::string& descrName = "SaliencyMapService",
        const std::string& tagName = "SaliencyMapService");

  ~SaliencyMapI();

  //This is were the visual cortex outputp message will be received
  virtual void evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&);

  void initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx); //Init the SimEvents by attaching to a topic and getting a messagePrx
  void unsubscribeSimEvents();


private:
  SimEvents::EventsPrx itsEventsPub; //the Event publisher
  nub::soft_ref<EnvSaliencyMap> itsSMap;
  std::vector<TopicInfo> itsTopicsSubscriptions; //out subscription to the various topics
  Ice::ObjectPrx itsObjectPrx; //Our proxy
  Image<byte> itsBiasImg;


};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // NEURO_ENVSaliencyMap_H_DEFINED
