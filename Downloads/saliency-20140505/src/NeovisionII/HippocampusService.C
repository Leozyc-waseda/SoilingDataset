/*!@file NeovisionII/HippocampusService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/HippocampusService.C $
// $Id: HippocampusService.C 13901 2010-09-09 15:12:26Z lior $
//

#include "NeovisionII/HippocampusService.H"
#include "Util/sformat.H"
#include "Image/DrawOps.H"

using namespace std;
using namespace ImageIceMod;

HippocampusI::HippocampusI(OptionManager& mgr,
    nub::ref<OutputFrameSeries> ofs,
    const std::string& descrName,
    const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsOfs(ofs),
  itsCameraDistance(880.954432),
  itsCameraAngle(2.321289),
  itsRadPerTilt(0.001298016),
  itsRadPerPan(0.001824947)
{
  //Subscribe to the various topics
  IceStorm::TopicPrx topicPrx;
  itsTopicsSubscriptions.push_back(TopicInfo("PrefrontalCortexTopic", topicPrx));

  resetCurrentObject();
}



HippocampusI::~HippocampusI()
{
  unsubscribeSimEvents();
}

void HippocampusI::start2()
{
}

void HippocampusI::resetCurrentObject()
{
  itsCurrentObject.objName = "noname";
  itsCurrentObject.loc = Point2D<int>(-1, -1);
  itsCurrentObject.size = 0;
}

void HippocampusI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a Segmenter Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("HippocampusTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("HippocampusTopic"); //The retina topic does not exists, create
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
      LFATAL("Error! No %s topic found!", itsTopicsSubscriptions[i].name.c_str());
    } catch (const char* msg) {
      LINFO("Error %s", msg);
    } catch (const Ice::Exception& e) {
      cerr << e << endl;
    }
  }
}

void HippocampusI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


void HippocampusI::evolve(const SimEvents::EventMessagePtr& eMsg,
    const Ice::Current&)
{

  if (eMsg->ice_isA("::SimEvents::HippocampusBiasMessage")){
    SimEvents::HippocampusBiasMessagePtr msg = SimEvents::HippocampusBiasMessagePtr::dynamicCast(eMsg);

    LINFO("Got bias message objName:%s loc:%ix%i rotation=%0.2f",
        msg->objectName.c_str(), msg->loc.i, msg->loc.j, msg->rotation*M_PI);

    //Get the image
    Image<PixRGB<byte> > retinaImg(320,240,ZEROS);

    if (msg->img.width > 0 && msg->img.height > 0)
      retinaImg = Ice2Image<PixRGB<byte> >(msg->img);

    if (msg->objectName == "MODE:rec")
    {
      itsCurrentObject.loc = Point2D<int>(msg->loc.i, msg->loc.j);
      itsCurrentObject.size = 20;
      itsCurrentObject.objName = "MODE:rec";
    }

    if (itsCurrentObject.loc.isValid() && msg->objectName != "nomatch" && msg->loc.i == -2 && msg->loc.j == -2)
    {
      itsCurrentObject.objName = msg->objectName;
      itsCurrentObject.pos = get3Dpos(msg->pan, msg->tilt);
      itsCurrentObject.rotation = msg->rotation;
      //Update the state
      itsObjectsState[itsCurrentObject.objName] = itsCurrentObject;
    }

    //Show the current objects;

    Image<PixRGB<byte> > stateMap = retinaImg;

    if (itsCurrentObject.objName != "nomatch")
    {
      if (itsCurrentObject.objName == "MODE:rec")
        drawCircle(stateMap, itsCurrentObject.loc, (int)itsCurrentObject.size, PixRGB<byte>(255,0,0),2);
      else
      {
        drawCircle(stateMap, itsCurrentObject.loc, (int)itsCurrentObject.size, PixRGB<byte>(0,255,0),2);
        writeText(stateMap, itsCurrentObject.loc, itsCurrentObject.objName.c_str(),
            PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));
      }
    }

    Image<byte> trmMap(stateMap.getDims(), ZEROS);
    trmMap.clear(0);

    Image<PixRGB<byte> > STMMap(512, 512, ZEROS);

    //Default IOR
    drawFilledRect(trmMap, Rectangle(Point2D<int>(280,0), Dims(40, 230)), (byte)0);
    drawFilledRect(trmMap, Rectangle(Point2D<int>(0,0), Dims(40, 230)), (byte)0);

    if (itsObjectsState.size() > 0)
    {
      SimEvents::HippocampusMessagePtr hMsg = new SimEvents::HippocampusMessage;
      SimEvents::ObjectsStateSeq objectsStateSeq;

      //Show the current objects
      for( std::map<std::string, ObjectState>::iterator itr = itsObjectsState.begin(); itr != itsObjectsState.end(); itr++)
      {
        LINFO("Object %s",  itr->second.objName.c_str());
        drawCircle(stateMap, itr->second.loc, (int)itr->second.size, PixRGB<byte>(0,255,0),2);
        writeText(stateMap, itr->second.loc, itr->second.objName.c_str(),
            PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));

        //Build the task relevent map
        //drawDisk(trmMap, itr->second.loc, (int)itr->second.size+10, (byte)0);
        Point2D<int> loc = itr->second.loc;
        int length = trmMap.getWidth()-(loc.i+30);
        drawFilledRect(trmMap, Rectangle(Point2D<int>(loc.i+30,loc.j-20), Dims(length, 40)), (byte)1);

        //Add the object the seq

        drawObject(STMMap, itr->second);

        //Check if object is in danger

        std::string msg;
        bool inDanger = false;
        for( std::map<std::string, ObjectState>::iterator itr2 = itsObjectsState.begin(); itr2 != itsObjectsState.end(); itr2++)
        {
          if (itr->second.pos.x > itr2->second.pos.x && itr->second.pos.x < itr2->second.pos.x+1000 &&
              itr->second.pos.y > itr2->second.pos.y-30 && itr->second.pos.y < itr2->second.pos.y+30)
          {
            msg = itr->second.objName + " is threatened by " + itr2->second.objName;
            inDanger = true;
            LINFO("%s", msg.c_str());
            writeText(STMMap, Point2D<int>(100,100), msg.c_str(),
                PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));
          }
        }

        SimEvents::ObjectState objectState;
        objectState.objectName = itr->second.objName;
        objectState.objectMsg = msg;
        objectState.loc.i = itr->second.loc.i;
        objectState.loc.j = itr->second.loc.j;
        objectState.pos.x = itr->second.pos.x;
        objectState.pos.y = itr->second.pos.y;
        objectState.pos.z = itr->second.pos.z;
        objectState.inDanger = inDanger;
        objectState.rotation = itr->second.rotation;
        objectState.size = itr->second.size;
        objectsStateSeq.push_back(objectState);

      }
      hMsg->objectsState = objectsStateSeq;
      itsEventsPub->evolve(hMsg);
    }

    //send the trm map
    SimEvents::SaliencyMapBiasMessagePtr sMapBiasMsg = new SimEvents::SaliencyMapBiasMessage;
    sMapBiasMsg->biasImg = Image2Ice(trmMap);
    itsEventsPub->evolve(sMapBiasMsg);


    itsOfs->writeRGB(stateMap, "Hippocampus", FrameInfo("Hippocampus", SRC_POS));
    //itsOfs->writeRGB(STMMap, "STMMap", FrameInfo("STMMap", SRC_POS));
    itsOfs->updateNext();


  }

}

void HippocampusI::drawObject(Image<PixRGB<byte> > & img, ObjectState &objState)
{
  Point2D<int> loc( (img.getWidth()/2) + objState.pos.x, img.getHeight() - objState.pos.y);
  drawCircle(img, loc,
      (int)objState.size, PixRGB<byte>(0,255,0),2);
  writeText(img, loc, objState.objName.c_str(),
      PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));
  drawLine(img, loc,
      objState.rotation, objState.size*3, PixRGB<byte>(255,0,0));

  if (objState.objName == "soldier")
  {
    int length = img.getWidth()-loc.i;
    drawRect(img, Rectangle(Point2D<int>(loc.i,loc.j-30), Dims(length, 60)), PixRGB<byte>(0,255,0));
  }
}

Point3D<float> HippocampusI::get3Dpos(int pan, int tilt)
{
  Point3D<float> pos;

  float cameraHeight = cos(M_PI-itsCameraAngle)*itsCameraDistance;
  float tiltAng = (M_PI-itsCameraAngle)+tilt*itsRadPerTilt;
  float panAng = pan*itsRadPerPan;

  pos.y = tan(tiltAng)*cameraHeight;
  pos.x = tan(panAng)*sqrt( (cameraHeight*cameraHeight) + (pos.y*pos.y));
  pos.z = 35;
  LINFO("Converting pan=%i tilt=%i to x=%f y=%f", pan, tilt, pos.x, pos.y);

  return pos;
}


/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class HippocampusService : public Ice::Service {
  protected:
    virtual bool start(int, char* argv[]);
    virtual bool stop() {
      if (itsMgr)
        delete itsMgr;
      return true;
    }

  private:
    Ice::ObjectAdapterPtr itsAdapter;
    ModelManager *itsMgr;
};

bool HippocampusService::start(int argc, char* argv[])
{

  itsMgr = new ModelManager("HippocampusService");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*itsMgr));
  itsMgr->addSubComponent(ofs);

  nub::ref<HippocampusI> hipp(new HippocampusI(*itsMgr, ofs));
  itsMgr->addSubComponent(hipp);

  itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::HippocampusPort);
  itsAdapter = communicator()->createObjectAdapterWithEndpoints("HippocampusAdapter",
      adapterStr);

  Ice::ObjectPtr object = hipp.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("Hippocampus"));
  hipp->initSimEvents(communicator(), objectPrx);
  itsAdapter->activate();

  itsMgr->start();

  return true;
}

// ######################################################################
int main(int argc, char** argv) {

  HippocampusService svc;
  return svc.main(argc, argv);
}
