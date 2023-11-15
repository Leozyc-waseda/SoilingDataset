/*!@file NeovisionII/MissileLauncher.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/MissileLauncher.H $
// $Id: MissileLauncher.H 13900 2010-09-09 14:48:32Z lior $
//

#ifndef NEOVISIONII_MissileLauncher_SERVICE
#define NEOVISIONII_MissileLauncher_SERVICE

#include <Ice/Ice.h>
#include <Ice/Service.h>
#include <IceStorm/IceStorm.h>
#include <IceUtil/Thread.h>
#include "Component/ModelManager.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"

#include "Ice/BrainObjects.ice.H"
#include "Ice/SimEvents.ice.H"
#include "Ice/IceImageUtils.H"
#include "Ice/ImageIce.ice.H"

#ifdef HAVE_USB_H
#include <usb.h>
#endif

using namespace std;
using namespace ImageIceMod;

#define COUNTDOWN 80

class MissileLauncher : public ModelComponent, public BrainObjects::SimulationViewer, public IceUtil::Thread {
  public:

    struct TopicInfo
    {
      TopicInfo(const char* _name, IceStorm::TopicPrx& prx) :
        name(_name), topicPrx(prx)
      {}
      std::string name;
      IceStorm::TopicPrx topicPrx;
    };


    MissileLauncher(ModelManager& mgr,
        nub::ref<OutputFrameSeries> ofs,
        const std::string& descrName = "MissileLauncher",
        const std::string& tagName = "MissileLauncher");

    ~MissileLauncher();


    //This is were the evolve message will be received
    virtual void evolve(const SimEvents::EventMessagePtr& eMsg,
        const Ice::Current&);

    void initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx object);
    void unsubscribeSimEvents();

    void send_command_cheeky(const char *cmd);


    virtual void run();

  private:
    nub::ref<OutputFrameSeries> itsOfs;
    std::vector<TopicInfo> itsTopicsSubscriptions; //out subscription to the various topics
    Ice::ObjectPrx itsObjectPrx; //Our proxy
    Image<PixRGB<byte> > itsMarkupImg;
    Image<PixRGB<byte> > itsRetinaImg;
    Image<byte> itsVCX;
    Image<byte> itsSMap;
    Image<PixRGB<byte> > itsSegImg;

    Point2D<int> itsTrackerLoc;

#ifdef HAVE_USB_H
    struct usb_bus *itsBusses;
    struct usb_device *itsDev;
    usb_dev_handle *itsLauncher;
#endif

    int itsCountdown;

};

#endif
