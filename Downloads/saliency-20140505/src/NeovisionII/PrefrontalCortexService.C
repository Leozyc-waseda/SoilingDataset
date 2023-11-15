/*!@file Neuro/PrefrontalCortexService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/PrefrontalCortexService.C $
// $Id: PrefrontalCortexService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/PrefrontalCortexService.H"
#include "Image/ColorOps.H"

PrefrontalCortexI::PrefrontalCortexI(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName),
  itsCurrentState(0),
  itsArmXOffset(358.76), //arm offset from camera
  itsArmYOffset(-372.16),
  itsArmZOffset(-80.00)
{
  //Subscribe to the various topics
  IceStorm::TopicPrx topicPrx;
  itsTopicsSubscriptions.push_back(TopicInfo("RetinaTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("SaliencyMapTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("VisualTrackerTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("SegmenterTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("CameraCtrlTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("InferotemporalCortexTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("HippocampusTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("GUITopic", topicPrx));

  assignStates();
}

PrefrontalCortexI::~PrefrontalCortexI()
{
  unsubscribeSimEvents();
}

void PrefrontalCortexI::init()
{
  SimEvents::GUIInputMessagePtr ggMsg = new SimEvents::GUIInputMessage;
  ggMsg->msg = "Please place my pieces and yours on the table and press enter.";
  ggMsg->rollDice = false;
  itsEventsPub->evolve(ggMsg);
  //itsCurrentState = WAIT_FOR_GAME_START;
  itsCurrentState = INIT_PTZ;

  getUserInput = true;
}

void PrefrontalCortexI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a PrefrontalCortex Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("PrefrontalCortexTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("PrefrontalCortexTopic"); //The retina topic does not exists, create
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
    } catch (const char* msg) {
      LINFO("Error %s", msg);
    } catch (const Ice::Exception& e) {
      cerr << e << endl;
    }
  }
}

void PrefrontalCortexI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


void PrefrontalCortexI::evolve(const SimEvents::EventMessagePtr& eMsg,
    const Ice::Current&)
{

  if(eMsg->ice_isA("::SimEvents::RetinaMessage")){
    SimEvents::RetinaMessagePtr msg = SimEvents::RetinaMessagePtr::dynamicCast(eMsg);
    itsCurrentRetinaImg = Ice2Image<PixRGB<byte> >(msg->img);
  }

  int nextState = processState(itsCurrentState, eMsg);
  setCurrentState(nextState);
}

void PrefrontalCortexI::setSalientTrackLoc(const SimEvents::SaliencyMapMessagePtr& msg)
{
  //Assign tracker to the most salient locations
  SimEvents::VisualTrackerBiasMessagePtr bMsg = new SimEvents::VisualTrackerBiasMessage;
  for(uint i=0; i<msg->nMostSalientLoc.size(); i++)
  {
    SimEvents::LocInfo locInfo = msg->nMostSalientLoc[i];
    LINFO("Assigning the most salient location to a track point: %ix%i",
        locInfo.fullresMaxpos.i, locInfo.fullresMaxpos.j);
    SimEvents::TrackInfo trackInfo;
    trackInfo.pos.i = locInfo.fullresMaxpos.i;
    trackInfo.pos.j = locInfo.fullresMaxpos.j;
    trackInfo.err = 0;
    bMsg->locToTrack.push_back(trackInfo);

    //Send a message to the hippocampus that we are going to rec this location
    SimEvents::HippocampusBiasMessagePtr hMsg = new SimEvents::HippocampusBiasMessage;
    hMsg->loc = trackInfo.pos;
    hMsg->objectName = "MODE:rec";
    hMsg->img = Image2Ice(itsCurrentRetinaImg);
    itsEventsPub->evolve(hMsg);

    break; //For now track the most salient location
  }
  itsEventsPub->evolve(bMsg);
}

void PrefrontalCortexI::setTrackLoc(Point2D<int> loc)
{
  //Assign tracker to the most salient locations
  SimEvents::VisualTrackerBiasMessagePtr bMsg = new SimEvents::VisualTrackerBiasMessage;
  LINFO("Assigning the location to a track point: %ix%i", loc.i, loc.j);
  SimEvents::TrackInfo trackInfo;
  trackInfo.pos.i = loc.i;
  trackInfo.pos.j = loc.j;
  trackInfo.err = 0;
  bMsg->locToTrack.push_back(trackInfo);

  //Send a message to the hippocampus that we are going to rec this location
  SimEvents::HippocampusBiasMessagePtr hMsg = new SimEvents::HippocampusBiasMessage;
  hMsg->loc = trackInfo.pos;
  hMsg->objectName = "MODE:rec";
  hMsg->img = Image2Ice(itsCurrentRetinaImg);
  itsEventsPub->evolve(hMsg);
  itsEventsPub->evolve(bMsg);

}

void PrefrontalCortexI::setTrackSegLoc(const SimEvents::VisualTrackerMessagePtr& msg)
{
  SimEvents::SegmenterBiasMessagePtr sMsg = new SimEvents::SegmenterBiasMessage;
  for(uint i=0; i<msg->trackLocs.size(); i++)
  {
    SimEvents::TrackInfo trackInfo = msg->trackLocs[i];
    //LINFO("Assigning the tracker location to a segment point: %ix%i",
    //    trackInfo.pos.i, trackInfo.pos.j);
    SimEvents::SegInfo segInfo;
    segInfo.loc.i = trackInfo.pos.i;
    segInfo.loc.j = trackInfo.pos.j;
    sMsg->regionsToSeg.push_back(segInfo);
    break; //For now seg only one location
  }
  itsEventsPub->evolve(sMsg);
}

void PrefrontalCortexI::setPTZoomIn()
{
  //Send a message that we want to ptz
  SimEvents::CameraCtrlBiasMessagePtr ptzMsg = new SimEvents::CameraCtrlBiasMessage;
  ptzMsg->zoomIn = true;
  LINFO("Sending Zooming in");
  itsEventsPub->evolve(ptzMsg);
}

int PrefrontalCortexI::processState(int state,  const SimEvents::EventMessagePtr& eMsg)
{
  if ((uint)state > itsStates.size())
  {
    LINFO("invalid state %i", state);
    return -1;
  }

  switch(state)
  {
    case WAIT_FOR_GAME_START:
      if (eMsg->ice_isA("::SimEvents::GUIOutputMessage")){
        SimEvents::GUIOutputMessagePtr msg = SimEvents::GUIOutputMessagePtr::dynamicCast(eMsg);
        LINFO("Key %i", msg->key);

        if (msg->key == 36)
        {
          SimEvents::GUIInputMessagePtr ggMsg = new SimEvents::GUIInputMessage;
          ggMsg->msg = "Please wait while I figure where the pieces are.";
          ggMsg->rollDice = false;
          itsEventsPub->evolve(ggMsg);
          return itsStates[state].nextState;
        }
      }
      return WAIT_FOR_GAME_START;


      // Get the game state
    case INIT_PTZ:
      {
        LINFO("INIT PTZ");
        //Init the ptz
        SimEvents::CameraCtrlBiasMessagePtr ptzMsg = new SimEvents::CameraCtrlBiasMessage;
        ptzMsg->initPtz = true;
        ptzMsg->zoomIn = false;
        ptzMsg->zoomOut = false;
        itsEventsPub->evolve(ptzMsg);
        return itsStates[state].nextState;
      }
    case INIT_PTZ_DONE: //Check for ptz init done
      //check if we finished initlizing the ptz
      if (eMsg->ice_isA("::SimEvents::CameraCtrlMessage")){
        SimEvents::CameraCtrlMessagePtr msg = SimEvents::CameraCtrlMessagePtr::dynamicCast(eMsg);
        if (msg->initPtzDone)
        {
          //Send the image to hippocampus to check if anything has changed
          return itsStates[state].nextState;
        }
      }
      return INIT_PTZ_DONE;
    case SET_TRACKING: //Set tracking
      if (!getUserInput)
      {
        if (eMsg->ice_isA("::SimEvents::SaliencyMapMessage")){
          SimEvents::SaliencyMapMessagePtr msg = SimEvents::SaliencyMapMessagePtr::dynamicCast(eMsg);
          setSalientTrackLoc(msg);
          setPTZoomIn();
          getUserInput = true;
          return itsStates[state].nextState;
        }
      } else {

        if (eMsg->ice_isA("::SimEvents::GUIOutputMessage")){
          SimEvents::GUIOutputMessagePtr msg = SimEvents::GUIOutputMessagePtr::dynamicCast(eMsg);
          if (msg->loc.i > 0 && msg->loc.j > 0)
          {
            setTrackLoc(Point2D<int>(msg->loc.i, msg->loc.j));
            setPTZoomIn();
            //getUserInput = false;
            return itsStates[state].nextState;
          }
        }
      }

      return SET_TRACKING;
    case TRACKING: //Tracking wait for done
      {
        if (eMsg->ice_isA("::SimEvents::CameraCtrlMessage")){
          SimEvents::CameraCtrlMessagePtr msg = SimEvents::CameraCtrlMessagePtr::dynamicCast(eMsg);
          if (msg->zoomDone)
          {
            itsPan = msg->pan;
            itsTilt = msg->tilt;
            return itsStates[state].nextState;
          }
        }
      }
      return TRACKING;
    case SET_SEG_LOC:
      if (eMsg->ice_isA("::SimEvents::VisualTrackerMessage")){
        SimEvents::VisualTrackerMessagePtr msg = SimEvents::VisualTrackerMessagePtr::dynamicCast(eMsg);
        if (msg->trackLocs.size()) //we have a tracking point, continue tracking
        {
          setTrackSegLoc(msg);
          itsCount=0;
          return itsStates[state].nextState;
        }
      }
      return SET_SEG_LOC;

    case GET_SEG_IMG:  //Wait for an img to recognize
      if(eMsg->ice_isA("::SimEvents::SegmenterMessage")){
        SimEvents::SegmenterMessagePtr msg = SimEvents::SegmenterMessagePtr::dynamicCast(eMsg);

        SimEvents::InfrotemporalCortexBiasMessagePtr sMsg = new SimEvents::InfrotemporalCortexBiasMessage;
        for(uint i=0; i<msg->segLocs.size(); i++)
        {
          SimEvents::SegInfo segInfo = msg->segLocs[i];
          sMsg->img = segInfo.img;
        }
        LINFO("Send img to SIFT");

        itsEventsPub->evolve(sMsg);
        return itsStates[state].nextState;
      }
      return GET_SEG_IMG;

    case GET_OBJ_INFO:  //Done
      //Wait for rec. message
      if(eMsg->ice_isA("::SimEvents::InfrotemporalCortexMessage")){
        SimEvents::InfrotemporalCortexMessagePtr msg = SimEvents::InfrotemporalCortexMessagePtr::dynamicCast(eMsg);
        LINFO("Got message");

        SimEvents::HippocampusBiasMessagePtr hMsg = new SimEvents::HippocampusBiasMessage;
        hMsg->loc.i = -2; hMsg->loc.j = -2;
        hMsg->pan = itsPan;
        hMsg->tilt = itsTilt;
        hMsg->rotation = msg->rotation;

        hMsg->objectName = msg->objectName;
        hMsg->img.width=0; hMsg->img.height=0;

        itsEventsPub->evolve(hMsg);

        LINFO("Got object %s with score %f",
            msg->objectName.c_str(), msg->confidence);
        if (msg->objectName == "nomatch" && itsCount < 3)
        {
          itsCount++;
          return GET_SEG_IMG;
        }
        else
          return itsStates[state].nextState;
      }
      return GET_OBJ_INFO;

    case GET_OBJ_STATE:
      if(eMsg->ice_isA("::SimEvents::HippocampusMessage")){
        SimEvents::HippocampusMessagePtr hMsg = SimEvents::HippocampusMessagePtr::dynamicCast(eMsg);

        //Show the objects we got so far
        for(uint i=0; i<hMsg->objectsState.size(); i++)
        {
          SimEvents::ObjectState objectState = hMsg->objectsState[i];
          LINFO("Got object %s at %ix%i (%f,%f,%f) inDanger=%i",
              objectState.objectName.c_str(),
              objectState.loc.i, objectState.loc.j,
              objectState.pos.x, objectState.pos.y, objectState.pos.z,
              objectState.inDanger);

          //if (objectState.inDanger)
          {
            //Move object out of the way
            LINFO("Moving object %s", objectState.objectName.c_str());
            SimEvents::PrimaryMotorCortexBiasMessagePtr msg = new SimEvents::PrimaryMotorCortexBiasMessage;
            msg->moveArm = true;
            msg->armPos.x = objectState.pos.x + itsArmXOffset;
            msg->armPos.y = objectState.pos.y + itsArmYOffset;
            msg->armPos.z = itsArmZOffset;
            msg->armPos.rot = 0;
            msg->armPos.roll = 0;
            msg->armPos.gripper = 0;

            itsEventsPub->evolve(msg);
          }
        }

        //Check if we got all the objects
        if (hMsg->objectsState.size() == 4)
          return ZOOM_OUT; //play the game
        else
          return itsStates[state].nextState;
      }
      return GET_OBJ_STATE;


    case ZOOM_OUT:
      {
        //Init the ptz
        SimEvents::CameraCtrlBiasMessagePtr ptzMsg = new SimEvents::CameraCtrlBiasMessage;
        ptzMsg->initPtz = true;
        ptzMsg->zoomIn = false;
        ptzMsg->zoomOut = false;
        itsEventsPub->evolve(ptzMsg);
        return itsStates[state].nextState;
      }
    case ZOOM_OUT_DONE:
      //check if we finished initlizing the ptz
      if (eMsg->ice_isA("::SimEvents::CameraCtrlMessage")){
        SimEvents::CameraCtrlMessagePtr msg = SimEvents::CameraCtrlMessagePtr::dynamicCast(eMsg);
        if (msg->initPtzDone)
        {
          //Send the image to hippocampus to check if anything has changed

          return itsStates[state].nextState;
        }
      }
      return ZOOM_OUT_DONE;

    case GAMEPLAY_PLAYERMOVE:
      {
        SimEvents::GUIInputMessagePtr ggMsg = new SimEvents::GUIInputMessage;
        ggMsg->msg = "Please move your pieces and press enter.";
        ggMsg->rollDice = false;
        itsEventsPub->evolve(ggMsg);
      }
      return itsStates[state].nextState;

    case GAMEPLAY_PLAYERMOVE_DONE:
      if (eMsg->ice_isA("::SimEvents::GUIOutputMessage")){
        SimEvents::GUIOutputMessagePtr msg = SimEvents::GUIOutputMessagePtr::dynamicCast(eMsg);

        if (msg->key == 36)
        {
          SimEvents::GUIInputMessagePtr ggMsg = new SimEvents::GUIInputMessage;
          ggMsg->msg = "Please wait while I figure where the pieces are.";
          ggMsg->rollDice = false;
          itsEventsPub->evolve(ggMsg);
          return itsStates[state].nextState;
        }
      }
      return GAMEPLAY_PLAYERMOVE_DONE;
  }


  return -1; //Return an invalid state
}



void PrefrontalCortexI::assignStates()
{
  itsStates.push_back(StateInfo(START_OF_GAME_MSG, "Wait for Game to start",
        WAIT_FOR_GAME_START));
  itsStates.push_back(StateInfo(WAIT_FOR_GAME_START, "Start the Game", INIT_PTZ));
  itsStates.push_back(StateInfo(INIT_PTZ, "Init PTZ", INIT_PTZ_DONE));
  itsStates.push_back(StateInfo(INIT_PTZ_DONE, "Check PTZ Init Done", SET_TRACKING));
  itsStates.push_back(StateInfo(SET_TRACKING, "SetTracking", TRACKING));
  itsStates.push_back(StateInfo(TRACKING, "Tracking", SET_SEG_LOC));
  itsStates.push_back(StateInfo(SET_SEG_LOC, "Set SEgmantation location", GET_SEG_IMG));
  itsStates.push_back(StateInfo(GET_SEG_IMG, "Get Image to Recognize", GET_OBJ_INFO));
  itsStates.push_back(StateInfo(GET_OBJ_INFO, "Recognize Image", GET_OBJ_STATE));
  itsStates.push_back(StateInfo(GET_OBJ_STATE, "Get Game State", INIT_PTZ));
  itsStates.push_back(StateInfo(ZOOM_OUT, "Observe the game", ZOOM_OUT_DONE));
  itsStates.push_back(StateInfo(ZOOM_OUT_DONE, "Observe the game Done", GAMEPLAY_PLAYERMOVE));
  itsStates.push_back(StateInfo(GAMEPLAY_PLAYERMOVE, "Player move", GAMEPLAY_PLAYERMOVE_DONE));
  itsStates.push_back(StateInfo(GAMEPLAY_PLAYERMOVE_DONE, "Player move done", GAMEPLAY_PLAYERMOVE_DONE));

  setCurrentState(0);
}

void PrefrontalCortexI::setCurrentState(int s)
{
  if (s == -1) return;
  if ((uint)s > itsStates.size())
  {
    LINFO("Entering an invalid state %i", s);
    return;
  }
  if (itsCurrentState != s)
  {
    itsLastState = s;
    LINFO("Entering state: %i:%s", s, itsStates[s].name.c_str());
    itsCurrentState = s;

  }
}


/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class PrefrontalCortexService : public Ice::Service {
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

bool PrefrontalCortexService::start(int argc, char* argv[])
{

  itsMgr = new ModelManager("PrefrontalCortexService");

  nub::ref<PrefrontalCortexI> pc(new PrefrontalCortexI(*itsMgr));
  itsMgr->addSubComponent(pc);

  itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::PrefrontalCortexPort);
  itsAdapter = communicator()->createObjectAdapterWithEndpoints("PrefrontalCortexAdapter",
      adapterStr);

  Ice::ObjectPtr object = pc.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("PrefrontalCortex"));
  pc->initSimEvents(communicator(), objectPrx);
  itsAdapter->activate();
  itsMgr->start();

  pc->init();

  return true;
}

// ######################################################################
int main(int argc, char** argv) {

  PrefrontalCortexService svc;
  return svc.main(argc, argv);
}


