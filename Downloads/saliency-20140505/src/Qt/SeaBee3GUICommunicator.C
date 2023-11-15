#ifndef SeaBee3GUICommunicator_C
#define SeaBee3GUICommunicator_C

#include "Qt/SeaBee3GUICommunicator.H"
#include "Ice/ImageIce.ice.H"
#include "Ice/IceImageUtils.H"
//#include "Robots/SeaBeeIII/Ice/SeaBeeMessages.ice.H"

SeaBee3GUICommunicator::SeaBee3GUICommunicator(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName) :
  RobotBrainComponent(mgr, descrName, tagName),
  itsGUIRegistered(false),
  itsFwdFrameCount(0),
  itsLastFwdFrameCount(0),
  itsDwnFrameCount(0),
  itsLastDwnFrameCount(0)
{
}

void SeaBee3GUICommunicator::registerTopics()
{
  registerSubscription("BeeStemMessageTopic");
  registerSubscription("XBox360RemoteControlMessageTopic");
  registerSubscription("RetinaMessageTopic");
  registerSubscription("VisionRectangleMessageTopic");
  registerSubscription("SalientPointMessageTopic");
  registerSubscription("MovementControllerMessageTopic");
  registerPublisher("CameraConfigTopic");
  registerPublisher("BeeStemConfigTopic");
  registerPublisher("SeaBeeStateConditionMessageTopic");
}

void SeaBee3GUICommunicator::evolve()
{
  itsImageMutex.lock();
  Image<PixRGB<byte> > fwdImg = itsFwdImg;
  Image<PixRGB<byte> > dwnImg = itsDwnImg;
  itsImageMutex.unlock();

  itsGUIFormMutex.lock();
  if(itsGUIRegistered)
    {
      if(fwdImg.initialized() && itsFwdFrameCount != itsLastFwdFrameCount)
        {
          itsLastFwdFrameCount = itsFwdFrameCount;
          itsGUIForm->setImage(fwdImg, "FwdCamera");
        }

      if(dwnImg.initialized() && itsDwnFrameCount != itsLastDwnFrameCount)
        {
          itsLastDwnFrameCount = itsDwnFrameCount;
          itsGUIForm->setImage(dwnImg, "DwnCamera");
        }
    }
  itsGUIFormMutex.unlock();
}

void SeaBee3GUICommunicator::toggleCamera(std::string cameraID, bool active) {

  RobotSimEvents::CameraConfigMessagePtr msg = new RobotSimEvents::CameraConfigMessage;
  msg->cameraID=cameraID;
  msg->active=active;

  this->publish("CameraConfigTopic", msg);
}

void SeaBee3GUICommunicator::updatePoseSettings(int updateSelect, int heading, int depth, int speed)
{

  RobotSimEvents::BeeStemConfigMessagePtr msg = new RobotSimEvents::BeeStemConfigMessage;
  msg->desiredHeading = heading;
  msg->desiredDepth = depth;
  msg->desiredSpeed = speed;
  msg->updateDesiredValue = updateSelect;

  msg->headingK = 0.0;
  msg->headingP = 0.0;
  msg->headingI = 0.0;
  msg->headingD = 0.0;
  msg->updateHeadingPID = false;

  msg->depthK = 0.0;
  msg->depthP = 0.0;
  msg->depthI = 0.0;
  msg->depthD = 0.0;
  msg->updateDepthPID = false;

  msg->enablePID = 0;
  msg->enableVal = 0;


  this->publish("BeeStemConfigTopic", msg);
}

void SeaBee3GUICommunicator::updatePID(int pidSelect, float k, float p, float i, float d)
{


  RobotSimEvents::BeeStemConfigMessagePtr msg = new RobotSimEvents::BeeStemConfigMessage;
  msg->desiredHeading = 0.0;
  msg->desiredDepth = 0.0;
  msg->desiredSpeed = 0.0;
  msg->updateDesiredValue = 0;

  msg->headingK = 0.0;
  msg->headingP = 0.0;
  msg->headingI = 0.0;
  msg->headingD = 0.0;
  msg->updateHeadingPID = false;

  msg->depthK = 0.0;
  msg->depthP = 0.0;
  msg->depthI = 0.0;
  msg->depthD = 0.0;
  msg->updateDepthPID = false;

  msg->enablePID = 0;
  msg->enableVal = 0;

  // if heading
  if(pidSelect == 1)
    {
      msg->headingK = k;
      msg->headingP = p;
      msg->headingI = i;
      msg->headingD = d;
      msg->updateHeadingPID = 1;
    }
  // if depth
  else if(pidSelect == 0)
    {
      msg->depthK = k;
      msg->depthP = p;
      msg->depthI = i;
      msg->depthD = d;
      msg->updateDepthPID = 1;
    }



  this->publish("BeeStemConfigTopic", msg);
}


// ######################################################################
void SeaBee3GUICommunicator::updateMessage(const RobotSimEvents::EventMessagePtr& eMsg,
    const Ice::Current&)
{
  //Get a retina message
  if(eMsg->ice_isA("::RobotSimEvents::RetinaMessage"))
  {
    RobotSimEvents::RetinaMessagePtr retinaMessage = RobotSimEvents::RetinaMessagePtr::dynamicCast(eMsg);
    Image<PixRGB<byte> > retinaImage = Ice2Image<PixRGB<byte> >(retinaMessage->img);

    itsImageMutex.lock();

    if(retinaMessage->cameraID == "FwdCamera")
      {
        itsFwdImg = retinaImage;
        itsFwdFrameCount++;
      }
    else
      {
        itsDwnImg = retinaImage;
        itsDwnFrameCount++;
      }

    itsImageMutex.unlock();
  }
  // Get a XBox360RemoteControl Message
  else if(eMsg->ice_isA("::RobotSimEvents::JoyStickControlMessage"))
  {
    RobotSimEvents::JoyStickControlMessagePtr msg = RobotSimEvents::JoyStickControlMessagePtr::dynamicCast(eMsg);

    itsGUIFormMutex.lock();
    if(itsGUIRegistered)
    {
      if(msg->axis >= 0)
        itsGUIForm->setJSAxis(msg->axis, msg->axisVal);
      else
        LINFO("Button[%d] = %d",msg->button, msg->butVal);
    }
    itsGUIFormMutex.unlock();
  }
  // Get a BeeStem Message
  else if(eMsg->ice_isA("::RobotSimEvents::BeeStemMessage"))
  {

    RobotSimEvents::BeeStemMessagePtr msg = RobotSimEvents::BeeStemMessagePtr::dynamicCast(eMsg);
    itsGUIFormMutex.lock();
    if(itsGUIRegistered)
      {
        int killSwitch = msg->killSwitch;
        killSwitch = killSwitch;
        itsGUIForm->setSensorValues(msg->compassHeading,
                                    msg->compassPitch,
                                    msg->compassRoll,
                                    msg->internalPressure,
                                    msg->externalPressure,
                                    msg->headingOutput,
                                    msg->depthOutput);
      }


    itsGUIForm->setThrusterMeters(
        0,
        (int)msg->thruster1,
        (int)msg->thruster2,
        (int)msg->thruster3,
        (int)msg->thruster4,
        (int)msg->thruster5,
        (int)msg->thruster6,
        0
        );



      itsGUIFormMutex.unlock();
  }
  else if(eMsg->ice_isA("::RobotSimEvents::VisionRectangleMessage"))
    {
      RobotSimEvents::VisionRectangleMessagePtr msg = RobotSimEvents::VisionRectangleMessagePtr::dynamicCast(eMsg);
      itsGUIFormMutex.lock();
      if(itsGUIRegistered)
        {
          if(msg->isFwdCamera)
            {
              for(uint i = 0; i < msg->quads.size(); i++)
                {
                  itsGUIForm->pushFwdRectangle(msg->quads[i]);
                }
            }
          else
            {
              for(uint i = 0; i < msg->quads.size(); i++)
                {

                  itsGUIForm->pushRectRatio(msg->quads[i].ratio);
                  itsGUIForm->pushRectAngle(msg->quads[i].angle);

                  itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].tl.i,msg->quads[i].tl.j));
                  itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].tr.i, msg->quads[i].tr.j));
                  itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].br.i, msg->quads[i].br.j));
                  itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].bl.i, msg->quads[i].bl.j));
                }
            }
        }
      itsGUIFormMutex.unlock();
    }
  else if(eMsg->ice_isA("::RobotSimEvents::SalientPointMessage"))
    {
      RobotSimEvents::SalientPointMessagePtr msg = RobotSimEvents::SalientPointMessagePtr::dynamicCast(eMsg);
      itsGUIFormMutex.lock();
      if(itsGUIRegistered)
        {
          itsGUIForm->setSalientPoint(Point2D<float>(msg->x,msg->y));
        }
      itsGUIFormMutex.unlock();
    }
  else if(eMsg->ice_isA("::RobotSimEvents::MovementControllerMessage"))
    {
      RobotSimEvents::MovementControllerMessagePtr msg = RobotSimEvents::MovementControllerMessagePtr::dynamicCast(eMsg);
      itsGUIFormMutex.lock();
      if(itsGUIRegistered)
        {
          itsGUIForm->setSensorVotes(msg->votes);
          itsGUIForm->setCompositeHeading(msg->compositeHeading);
          itsGUIForm->setCompositeDepth(msg->compositeDepth);
        }
      itsGUIFormMutex.unlock();
    }
}

// ######################################################################
void SeaBee3GUICommunicator::enablePID()
{
  RobotSimEvents::BeeStemConfigMessagePtr msg = new RobotSimEvents::BeeStemConfigMessage;
  msg->desiredHeading = 0.0;
  msg->desiredDepth = 0.0;
  msg->desiredSpeed = 0.0;
  msg->updateDesiredValue = 0;

  msg->headingK = 0.0;
  msg->headingP = 0.0;
  msg->headingI = 0.0;
  msg->headingD = 0.0;
  msg->updateHeadingPID = false;

  msg->depthK = 0.0;
  msg->depthP = 0.0;
  msg->depthI = 0.0;
  msg->depthD = 0.0;
  msg->updateDepthPID = false;

  msg->enablePID = 1;
  msg->enableVal = 1;

  this->publish("BeeStemConfigTopic", msg);
}

// ######################################################################
void SeaBee3GUICommunicator::disablePID()
{
  RobotSimEvents::BeeStemConfigMessagePtr msg = new RobotSimEvents::BeeStemConfigMessage;
  msg->desiredHeading = 0.0;
  msg->desiredDepth = 0.0;
  msg->desiredSpeed = 0.0;
  msg->updateDesiredValue = 0;

  msg->headingK = 0.0;
  msg->headingP = 0.0;
  msg->headingI = 0.0;
  msg->headingD = 0.0;
  msg->updateHeadingPID = false;

  msg->depthK = 0.0;
  msg->depthP = 0.0;
  msg->depthI = 0.0;
  msg->depthD = 0.0;
  msg->updateDepthPID = false;

  msg->enablePID = 1;
  msg->enableVal = 0;
  this->publish("BeeStemConfigTopic", msg);

}
// ######################################################################
void SeaBee3GUICommunicator::registerGUI(SeaBee3MainDisplayForm* form)
{
  itsGUIFormMutex.lock();
  itsGUIForm = form;
  itsGUIRegistered = true;
  itsGUIFormMutex.unlock();
}

void SeaBee3GUICommunicator::SeaBeeInjectorMsg(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j)
{
        RobotSimEvents::SeaBeeStateConditionMessagePtr msg;
        msg = new RobotSimEvents::SeaBeeStateConditionMessage;

        msg->InitDone = a;
        msg->GateFound = b;
        msg->GateDone = c;
        msg->ContourFoundFlare = d;
        msg->FlareDone = e;
        msg->ContourFoundBarbwire = f;
        msg->BarbwireDone = g;
        msg->ContourFoundBoxes = h;
        msg->BombingRunDone = i;
        msg->BriefcaseFound = j;
        msg->TimeUp = 0;

        publish("SeaBeeStateConditionMessageTopic",msg);
}
#endif

