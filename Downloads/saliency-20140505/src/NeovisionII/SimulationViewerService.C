/*!@file NeovisionII/SimulationViewerService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/SimulationViewerService.C $
// $Id: SimulationViewerService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/SimulationViewerService.H"
#include "Util/sformat.H"
#include "Image/DrawOps.H"
#include "Raster/Raster.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"
#include "GUI/DebugWin.H"
#include "Util/MathFunctions.H"


using namespace std;
using namespace ImageIceMod;

SimulationViewerI::SimulationViewerI(ModelManager& mgr,
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
  itsTopicsSubscriptions.push_back(TopicInfo("SegmenterTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("InferotemporalCortexTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("PrefrontalCortexTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("HippocampusTopic", topicPrx));

  itsDiceImg.reset(6);
  for(int i=0; i<6; i++)
  {
    char filename[255];
    sprintf(filename, "./etc/dice/120px-Dice-%i.png", i+1);
    itsDiceImg[i] = Raster::ReadRGB(filename);
  }

}

SimulationViewerI::~SimulationViewerI()
{
  unsubscribeSimEvents();
}


void SimulationViewerI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
    //Connect to the SimEvents
    Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
    IceStorm::TopicManagerPrx topicManager =
      IceStorm::TopicManagerPrx::checkedCast(obj);

    //Create a PrefrontalCortex Topic
    IceStorm::TopicPrx topic;
    try {
      topic = topicManager->retrieve("GUITopic"); //check if the Retina topic exists
    } catch (const IceStorm::NoSuchTopic&) {
      topic = topicManager->create("GUITopic"); //The retina topic does not exists, create
    }
    //Make a one way visualCortex message publisher for efficency
    Ice::ObjectPrx pub = topic->getPublisher()->ice_oneway();
    itsEventsPub = SimEvents::EventsPrx::uncheckedCast(pub);

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
      } catch (const Ice::Exception& e) {
        cerr << e << endl;
      }

    }
}

void SimulationViewerI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


void SimulationViewerI::evolve(const SimEvents::EventMessagePtr& eMsg,
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
      }
    }
  } else if (eMsg->ice_isA("::SimEvents::SegmenterMessage")){
    SimEvents::SegmenterMessagePtr msg = SimEvents::SegmenterMessagePtr::dynamicCast(eMsg);
    //draw the tracker locations
    if (itsMarkupImg.initialized())
    {
      for(uint i=0; i<msg->segLocs.size(); i++)
      {
        SimEvents::SegInfo segInfo = msg->segLocs[i];
        Rectangle segRect(Point2D<int>(segInfo.rect.tl.i, segInfo.rect.tl.j),
            Dims(segInfo.rect.br.i - segInfo.rect.tl.i,
              segInfo.rect.br.j - segInfo.rect.tl.j));
        if (segRect.isValid() && itsMarkupImg.rectangleOk(segRect))
        {
          drawRect(itsMarkupImg, segRect, PixRGB<byte>(255, 255, 0), 1);
        }
        itsSegImg = Ice2Image<PixRGB<byte> >(segInfo.img);
      }
    }
  } else if (eMsg->ice_isA("::SimEvents::InfrotemporalCortexMessage")){
    SimEvents::InfrotemporalCortexMessagePtr msg = SimEvents::InfrotemporalCortexMessagePtr::dynamicCast(eMsg);

    //draw the tracker locations
    if (itsMarkupImg.initialized())
    {
      const std::string txt =
        sformat("%s:%0.2f", msg->objectName.c_str(), msg->confidence);

      writeText(itsMarkupImg, Point2D<int>(msg->objLoc.i, msg->objLoc.j),
          txt.c_str(),
          PixRGB<byte>(255), PixRGB<byte>(0));
    }
  } else if (eMsg->ice_isA("::SimEvents::GUIInputMessage")){
    SimEvents::GUIInputMessagePtr msg = SimEvents::GUIInputMessagePtr::dynamicCast(eMsg);
    itsCurrentMsg = msg->msg;
  } else if(eMsg->ice_isA("::SimEvents::HippocampusMessage")){
        SimEvents::HippocampusMessagePtr hMsg = SimEvents::HippocampusMessagePtr::dynamicCast(eMsg);

        //Show the objects we got so far
        itsMutex.lock();
        itsObjectsState.clear();
        for(uint i=0; i<hMsg->objectsState.size(); i++)
        {
          SimEvents::ObjectState objectState = hMsg->objectsState[i];


          ObjectState objState;
          objState.objName =  objectState.objectName;
          objState.objMsg =  objectState.objectMsg;
          objState.loc.i =  objectState.loc.i;
          objState.loc.j =  objectState.loc.j;
          objState.pos.x =  objectState.pos.x/2;
          objState.pos.y =  objectState.pos.y/2;
          objState.pos.z =  objectState.pos.z;
          objState.rotation =  objectState.rotation;
          objState.size = objectState.size;
          itsObjectsState.push_back(objState);
        }
        itsMutex.unlock();
  }


}

int SimulationViewerI::getKey(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("GameInterface")
    : rutz::shared_ptr<XWinManaged>();

  if (uiwin.is_valid())
    return uiwin->getLastKeyPress();
  else
    return -1;
}

Point2D<int> SimulationViewerI::getMouseClick(nub::ref<OutputFrameSeries> &ofs, const char* wname)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow(wname)
    : rutz::shared_ptr<XWinManaged>();

  if (uiwin.is_valid())
    return uiwin->getLastMouseClick();
  else
    return Point2D<int>(-1,-1);
}


void SimulationViewerI::run()
{
  initRandomNumbers();
  int rollDice=0;

  Image<PixRGB<byte> > gameInterface(640,240,ZEROS);
  gameInterface.clear(PixRGB<byte>(255,255,255));
  inplacePaste(gameInterface, itsDiceImg[5], Point2D<int>(320/2-(itsDiceImg[5].getWidth()/2),0));

  while(1)
  {
    Layout<PixRGB<byte> > outDisp;

    if (itsMarkupImg.initialized())
      itsOfs->writeRGB(itsMarkupImg, "Markup", FrameInfo("Markup", SRC_POS));

    Point2D<int> clickLoc = getMouseClick(itsOfs, "Markup");
    if (clickLoc.isValid())
    {
      LINFO("Click at %i %i", clickLoc.i, clickLoc.j);
      SimEvents::GUIOutputMessagePtr goMsg = new SimEvents::GUIOutputMessage;
      goMsg->key = -1;
      goMsg->loc.i = clickLoc.i;
      goMsg->loc.j = clickLoc.j;
      goMsg->dice = -1;
      itsEventsPub->evolve(goMsg);
    }


    //if (itsSegImg.initialized())
    //  itsOfs->writeRGB(itsSegImg, "Segment", FrameInfo("Segment", SRC_POS));

    if (itsRetinaImg.initialized())
      outDisp = itsMarkupImg;

    if (itsVCX.initialized() && itsRetinaImg.initialized())
      outDisp =  vcat(outDisp, toRGB(rescale(itsVCX, itsRetinaImg.getDims())));

    if (itsSMap.initialized() &&
        itsRetinaImg.initialized() &&
        itsRetinaImg.getWidth() > 0 &&
        itsRetinaImg.getHeight() > 0 &&
        itsSMap.getWidth() > 0 &&
        itsSMap.getHeight() > 0 )
      outDisp = vcat(outDisp, toRGB(rescale(itsSMap, itsRetinaImg.getDims())));




    itsMutex.lock();
    std::vector<ObjectState> tmpObjects = itsObjectsState;
    itsMutex.unlock();

    Image<PixRGB<byte> > stmMap(512,512, ZEROS);
    for(uint i=0; i<tmpObjects.size(); i++)
                {
                        ObjectState objState = tmpObjects[i];

                        Point2D<int> loc( (stmMap.getWidth()/2) + objState.pos.x, stmMap.getHeight() - objState.pos.y);
                        drawCircle(stmMap, loc,
                                        (int)objState.size, PixRGB<byte>(0,255,0),2);
                        writeText(stmMap, loc, objState.objName.c_str(),
                                        PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));
                        drawLine(stmMap, loc,
                                        objState.rotation, objState.size*3, PixRGB<byte>(255,0,0));

                        if (objState.objName == "soldier")
                        {
                                int length = stmMap.getWidth()-loc.i;
                                drawRect(stmMap, Rectangle(Point2D<int>(loc.i,loc.j-30), Dims(length, 60)), PixRGB<byte>(0,255,0));
                        }

                        writeText(stmMap, Point2D<int>(0,0), objState.objMsg.c_str(),
                                        PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));

                }

                outDisp = hcat(outDisp, stmMap);


    if (outDisp.initialized())
                {
      itsOfs->writeRgbLayout(outDisp, "Output", FrameInfo("Output", SRC_POS));
                        itsOfs->updateNext();
                }


    if (rollDice)
    {
      int num = randomUpToIncluding(5);
      inplacePaste(gameInterface, itsDiceImg[num], Point2D<int>(320/2-(itsDiceImg[num].getWidth()/2),0));
      rollDice--;
      if (rollDice == 0)
        LINFO("Final Number %i", num+1);
    }

    if (itsCurrentMsg.size())
    {
      itsCurrentMsg += "                                     ";
      writeText(gameInterface, Point2D<int>(0, 200), itsCurrentMsg.c_str(),
          PixRGB<byte>(0,255,0), PixRGB<byte>(255,255,255));
      itsCurrentMsg.clear();
    }

    itsOfs->writeRGB(gameInterface, "GameInterface", FrameInfo("GameInterface", SRC_POS));

    int key = getKey(itsOfs);
    if (key != -1)
    {
      switch(key)
      {
        case 27:
          rollDice = 10;
          break;
        default:
          {
            SimEvents::GUIOutputMessagePtr goMsg = new SimEvents::GUIOutputMessage;
            goMsg->key = key;
            goMsg->loc.i = -1;
            goMsg->loc.j = -1;
            goMsg->dice = -1;
            itsEventsPub->evolve(goMsg);
          }
          break;
      }
    }

    usleep(10000);
  }
}


/////////////////////////// The SimulationViewer Service to init the retina and start as a deamon ///////////////
class SimulationViewerService : public Ice::Service {
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

bool SimulationViewerService::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("SimulationViewerService");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*itsMgr));
  itsMgr->addSubComponent(ofs);

  nub::ref<SimulationViewerI> simulationViewer(new SimulationViewerI(*itsMgr, ofs));
        itsMgr->addSubComponent(simulationViewer);

        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::SimulationViewerPort);
        itsAdapter = communicator()->createObjectAdapterWithEndpoints(
      "SimulationViewerAdapter", adapterStr);

        Ice::ObjectPtr object = simulationViewer.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("SimulationViewer"))->ice_oneway();

  simulationViewer->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

  //Start the simulation viewer thread
  IceUtil::ThreadPtr svThread = simulationViewer.get();
  svThread->start();
        return true;
}


// ######################################################################
int main(int argc, char** argv) {

  SimulationViewerService svc;
  return svc.main(argc, argv);
}

