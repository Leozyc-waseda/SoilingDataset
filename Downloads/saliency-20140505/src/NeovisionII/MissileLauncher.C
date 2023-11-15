/*!@file NeovisionII/MissileLauncher.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/MissileLauncher.C $
// $Id: MissileLauncher.C 13901 2010-09-09 15:12:26Z lior $
//

#include "NeovisionII/MissileLauncher.H"
#include "Util/sformat.H"
#include "Image/DrawOps.H"

using namespace std;
using namespace ImageIceMod;

MissileLauncher::MissileLauncher(ModelManager& mgr,
    nub::ref<OutputFrameSeries> ofs,
    const std::string& descrName,
    const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsOfs(ofs)
{
  //Subscribe to the various topics
  IceStorm::TopicPrx topicPrx;
  itsTopicsSubscriptions.push_back(TopicInfo("RetinaTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("VisualCortexTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("SaliencyMapTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("VisualTrackerTopic", topicPrx));


#ifdef HAVE_USB_H
  usb_init();
  usb_find_busses();
  usb_find_devices();

  itsBusses = usb_get_busses();

  struct usb_bus *bus = NULL;
  struct usb_device *dev = NULL;

  for (bus = itsBusses; bus && !dev; bus = bus->next) {
    for (dev = bus->devices; dev; dev = dev->next) {
      LINFO("Checking 0x%04x:0x%04x\n",
          dev->descriptor.idVendor,
          dev->descriptor.idProduct);
      if (dev->descriptor.idVendor == 0x0a81 &&
          dev->descriptor.idProduct == 0x0701) {
        itsDev = dev;
        LINFO("Found launcher");
        break;
      }
    }
  }

  if (!itsDev) {
    LFATAL("Unable to find device.\n");
  }


  itsLauncher = usb_open(itsDev);
  if (itsLauncher == NULL) {
    LFATAL("Unable to open device");
  }

  /* Detach kernel driver (usbhid) from device interface and claim */
  usb_detach_kernel_driver_np(itsLauncher, 0);
  usb_detach_kernel_driver_np(itsLauncher, 1);

  int ret = usb_set_configuration(itsLauncher, 1);
  if (ret < 0) {
    LFATAL("Unable to set device configuration");
  }
  ret = usb_claim_interface(itsLauncher, 0);
  if (ret < 0) {
    LFATAL("Unable to claim interface");
  }

  itsCountdown = COUNTDOWN;
#else
  LFATAL("Need libusb to work");
#endif
}



MissileLauncher::~MissileLauncher()
{
  unsubscribeSimEvents();
#ifdef HAVE_USB_H
  usb_release_interface(itsLauncher, 0);
  usb_close(itsLauncher);
#endif
}


void MissileLauncher::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Connect to the SimEvents
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Subscribe the SimulationViewer to theTopics
  itsObjectPrx = objectPrx;
  //subscribe  to the topics
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    try {
      IceStorm::QoS qos;
      itsTopicsSubscriptions[i].topicPrx =
        topicManager->retrieve(itsTopicsSubscriptions[i].name.c_str()); //Get the
      itsTopicsSubscriptions[i].topicPrx->subscribeAndGetPublisher(qos, itsObjectPrx); //Subscribe to the retina topic
    } catch (const IceStorm::NoSuchTopic&) {
      LINFO("Error! No %s topic found!", itsTopicsSubscriptions[i].name.c_str());
    }
  }
}

void MissileLauncher::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


void MissileLauncher::evolve(const SimEvents::EventMessagePtr& eMsg,
    const Ice::Current&)
{
  if(eMsg->ice_isA("::SimEvents::RetinaMessage")){
    SimEvents::RetinaMessagePtr msg = SimEvents::RetinaMessagePtr::dynamicCast(eMsg);
    itsRetinaImg = Ice2Image<PixRGB<byte> > (msg->img);
    itsMarkupImg = itsRetinaImg;
  } else if (eMsg->ice_isA("::SimEvents::VisualCortexMessage")){
    SimEvents::VisualCortexMessagePtr msg = SimEvents::VisualCortexMessagePtr::dynamicCast(eMsg);
    itsVCX = Ice2Image<byte>(msg->vco);
  } else if (eMsg->ice_isA("::SimEvents::SaliencyMapMessage")){
    SimEvents::SaliencyMapMessagePtr msg = SimEvents::SaliencyMapMessagePtr::dynamicCast(eMsg);
    itsSMap = Ice2Image<byte>(msg->smap);
    //draw the top N most salient points
    if (itsMarkupImg.initialized())
    {
      for(uint i=0; i<msg->nMostSalientLoc.size(); i++)
      {
        SimEvents::LocInfo locInfo = msg->nMostSalientLoc[i];

        if (i == 0)  //Draw the most salient location with a stronger color
        {
          drawRectSquareCorners(itsMarkupImg,
              Rectangle(Point2D<int>(locInfo.fullresMaxpos.i, locInfo.fullresMaxpos.j), Dims(3,3)),
              PixRGB<byte>(255, 0, 0), 6);
        } else {
          drawRectSquareCorners(itsMarkupImg,
              Rectangle(Point2D<int>(locInfo.fullresMaxpos.i, locInfo.fullresMaxpos.j), Dims(3,3)),
              PixRGB<byte>(150, 0, 0), 3);
        }
      }
    }
  } else if (eMsg->ice_isA("::SimEvents::VisualTrackerMessage")){
    SimEvents::VisualTrackerMessagePtr msg = SimEvents::VisualTrackerMessagePtr::dynamicCast(eMsg);
    //draw the tracker locations
    if (itsMarkupImg.initialized())
    {
      for(uint i=0; i<msg->trackLocs.size(); i++)
      {
        SimEvents::TrackInfo trackInfo = msg->trackLocs[i];
        drawRectSquareCorners(itsMarkupImg,
            Rectangle(Point2D<int>(trackInfo.pos.i, trackInfo.pos.j), Dims(3,3)),
            PixRGB<byte>(0, 255, 0), 3);

        itsTrackerLoc = Point2D<int>(trackInfo.pos.i, trackInfo.pos.j);
      }
    }
  }

}


/*
 * Command to control Dream Cheeky USB missile launcher
 */
void MissileLauncher::send_command_cheeky(const char *cmd)
{
#ifdef HAVE_USB_H
  char data[8];
  int ret;

  memset(data, 0, 8);
  if (!strcmp(cmd, "up")) {
    data[0] = 0x02;
  } else if (!strcmp(cmd, "down")) {
    data[0] = 0x01;
  } else if (!strcmp(cmd, "left")) {
    data[0] = 0x04;
  } else if (!strcmp(cmd, "right")) {
    data[0] = 0x08;
  } else if (!strcmp(cmd, "fire")) {
    data[0] = 0x10;
  } else if (!strcmp(cmd, "stop")) {
    data[0] = 0x20;
  } else {
    LINFO("Unknown command: %s", cmd);
  }

  ret = usb_control_msg(itsLauncher,
      USB_DT_HID,
      USB_REQ_SET_CONFIGURATION,
      USB_RECIP_ENDPOINT,
      0,
      data,
      8,                // Length of data.
      5000);                // Timeout
  if (ret != 8) {
    LINFO("Error: %s\n", usb_strerror());
  }
#endif
}


void MissileLauncher::run()
{
  while(1)
  {
    Layout<PixRGB<byte> > outDisp;

    if (itsMarkupImg.initialized())
      itsOfs->writeRGB(itsMarkupImg, "Markup", FrameInfo("Markup", SRC_POS));

    if (itsVCX.initialized() && itsRetinaImg.initialized())
      outDisp =  toRGB(rescale(itsVCX, itsRetinaImg.getDims()));

    if (outDisp.initialized())
      itsOfs->writeRgbLayout(outDisp, "Output", FrameInfo("Output", SRC_POS));


    if (itsMarkupImg.initialized())
    {
      LINFO("Tracking %ix%i", itsTrackerLoc.i, itsTrackerLoc.j);
      int errX = itsTrackerLoc.i - (itsMarkupImg.getWidth()/2);
      int errY = itsTrackerLoc.j - (itsMarkupImg.getHeight()/2);

      LINFO("Err %i,%i countdown=%i", errX, errY, itsCountdown);

      if (abs(errX) > 30)
      {
        if (errX > 0)
          send_command_cheeky("right");
        else
          send_command_cheeky("left");
        itsCountdown = COUNTDOWN;
      } else {

        itsCountdown--;
        if (abs(errY) > 20)
        {
          if (errY > 0)
            send_command_cheeky("down");
          else
            send_command_cheeky("up");
        } else {
          send_command_cheeky("stop");
        }
      }


      if (itsCountdown < 0)
      {
        LINFO("Shoot......");
        send_command_cheeky("fire");
        sleep(4);
        itsCountdown = COUNTDOWN;
      }
    }

    usleep(10000);
  }
}


/////////////////////////// The SimulationViewer Service to init the retina and start as a deamon ///////////////
class MissileLauncherService : public Ice::Service {
  protected:
    virtual bool start(int, char* argv[]);
    virtual bool stop()
    {
      if (itsMgr)
        delete itsMgr;
      return true;
    }
  private:
    Ice::ObjectAdapterPtr itsAdapter;
    ModelManager *itsMgr;
};

bool MissileLauncherService::start(int argc, char* argv[])
{

  itsMgr = new ModelManager("MissileLauncherServiceService");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*itsMgr));
  itsMgr->addSubComponent(ofs);

  nub::ref<MissileLauncher> missileLauncher(new MissileLauncher(*itsMgr, ofs));
  itsMgr->addSubComponent(missileLauncher);

  itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::SimulationViewerPort);
  itsAdapter = communicator()->createObjectAdapterWithEndpoints(
      "SimulationViewerAdapter", adapterStr);

  Ice::ObjectPtr object = missileLauncher.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("SimulationViewer"))->ice_oneway();

  missileLauncher->initSimEvents(communicator(), objectPrx);
  itsAdapter->activate();

  itsMgr->start();

  //Start the simulation viewer thread
  IceUtil::ThreadPtr svThread = missileLauncher.get();
  svThread->start();
  return true;
}


// ######################################################################
int main(int argc, char** argv) {

  MissileLauncherService svc;
  return svc.main(argc, argv);
}

