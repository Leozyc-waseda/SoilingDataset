#ifndef SeaBee3GUIIce_C
#define SeaBee3GUIIce_C

#include "Qt/SeaBee3GUIIce.H"
#include "Ice/ImageIce.ice.H"
#include "Ice/IceImageUtils.H"


#define MSG_BUFF_MAX 5

SeaBee3GUIIce::SeaBee3GUIIce(OptionManager& mgr,
                             const std::string& descrName,
                             const std::string& tagName) :
  RobotBrainComponent(mgr, descrName, tagName),
  itsGUIRegistered(false),
  itsFwdRetinaMsgCounter(0),
  itsDwnRetinaMsgCounter(0),
  itsOrangeSegEnabled(false),
  itsRedSegImagesSize(0),
  itsSalientPointsSize(0),
  itsSalientPointsEnabled(false),
  itsVisionMsgCounter(0),
  itsBeeStemMsgCounter(0),
  itsCompassMeter(140,130),
  itsDepthMeter(140,130),
  itsPressureMeter(100,90),
  itsCircleFillMeter(60,50,20),
  itsTargetLineMeter(100,50)
{
  itsTimer.reset();
  itsFwdVisionImage = Image<PixRGB<byte> >();
  itsDwnVisionImage = Image<PixRGB<byte> >();
}

void SeaBee3GUIIce::registerTopics()
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

void SeaBee3GUIIce::evolve()
{
  itsUpdateMutex.lock();
  if(itsGUIRegistered)
    {
      if(itsFwdRetinaImages.size() > 0)
        {
          updateFwdImg();

          if(itsOrangeSegImages.size() > 0 && itsOrangeSegEnabled)
            {
              Image<PixRGB<byte> > oj = itsOrangeSegImages.front();

              itsFwdVisionImage = oj;
              itsOrangeSegImages.pop_front();
            }

          if(itsSalientPoints.size() > 0 && itsSalientPointsEnabled)
            updateSaliencyImage();

          itsGUIForm->setFwdVisionImage(itsFwdVisionImage);

        }
      if(itsDwnRetinaImages.size() > 0)
        {
          updateDwnImg();

          itsGUIForm->setDwnVisionImage(itsDwnVisionImage);
        }

      if(itsBeeStemData.size() > 0)
        {
          updateBeeStemData();
        }
    }


  if(itsTimer.getSecs() >= 1.0)
    {
      itsGUIForm->setFwdRetinaMsgField(itsFwdRetinaMsgCounter);
      itsGUIForm->setDwnRetinaMsgField(itsDwnRetinaMsgCounter);
      itsGUIForm->setBeeStemMsgField(itsBeeStemMsgCounter);
      itsGUIForm->setVisionMsgField(itsVisionMsgCounter);
      Image<PixRGB<byte> > headingAxis = itsCircleFillMeter.render(itsFwdRetinaMsgCounter);
      Image<PixRGB<byte> > depthAxis = itsCircleFillMeter.render(itsDwnRetinaMsgCounter);
      Image<PixRGB<byte> > strafeAxis = itsCircleFillMeter.render(itsBeeStemMsgCounter);
      itsGUIForm->setAxesImages(headingAxis,depthAxis,strafeAxis);

      itsTimer.reset();
      itsFwdRetinaMsgCounter = 0;
      itsDwnRetinaMsgCounter = 0;
      itsBeeStemMsgCounter = 0;
      itsVisionMsgCounter = 0;
    }
  itsUpdateMutex.unlock();
}

// ######################################################################
void SeaBee3GUIIce::updateFwdImg()
{
  Image<PixRGB<byte> > img = itsFwdRetinaImages.front();
  itsGUIForm->setFwdImage(img);

  itsFwdVisionImage = img;

  itsFwdRetinaImages.pop_front();
}

// ######################################################################
void SeaBee3GUIIce::updateDwnImg()
{
  Image<PixRGB<byte> > img = itsDwnRetinaImages.front();
  itsGUIForm->setDwnImage(img);
  itsDwnVisionImage = img;

  itsDwnRetinaImages.pop_front();
}

// ######################################################################
void SeaBee3GUIIce::updateSaliencyImage()
{
  Point2D<int> pt = itsSalientPoints.front();

  //  drawTraj(itsFwdVisionImage,
  //&(itsSalientPoints[0]),&(itsSalientPoints[2]));

  //  LINFO("Point %d, %d\n",pt.i,pt.j);
  PixRGB<byte> color(0,0,0);
  drawCircle(itsFwdVisionImage, pt, 10, PixRGB<byte>(0,150,0), 1);
  drawCircle(itsFwdVisionImage, pt, 13, PixRGB<byte>(0,100,0), 1);
  drawCircle(itsFwdVisionImage, pt, 16, PixRGB<byte>(0,50,0), 1);
  drawCircle(itsFwdVisionImage, pt, 19, PixRGB<byte>(0,0,0), 1);


  drawDisk(itsFwdVisionImage, pt, 7, PixRGB<byte>(0,0,0));
  drawDisk(itsFwdVisionImage, pt, 4, PixRGB<byte>(0,255,0));

  itsSalientPoints.pop_front();
}

// ######################################################################
void SeaBee3GUIIce::updateBeeStemData()
{
  BeeStemData d = itsBeeStemData.front();

  Image<PixRGB<byte> > compassImg = itsCompassMeter.render(d.heading);
  Image<PixRGB<byte> > depthImg = itsDepthMeter.render(d.externalPressure);
  Image<PixRGB<byte> > pressureImg = itsPressureMeter.render(d.internalPressure);

  itsTargetLineMeter.setDesiredValue(d.desiredDepth);
  Image<PixRGB<byte> > depthPIDImg = itsTargetLineMeter.render(d.externalPressure);

  itsGUIForm->setCompassImage(compassImg);
  itsGUIForm->setDepthImage(depthImg);
  itsGUIForm->setPressureImage(pressureImg);
  itsGUIForm->setDepthPIDImage(depthPIDImg);

  itsGUIForm->setBeeStemData(d);

  itsBeeStemData.pop_front();
}

void SeaBee3GUIIce::setOrangeSegEnabled(bool enabled)
{
  itsUpdateMutex.lock();
  itsOrangeSegEnabled = enabled;
  itsUpdateMutex.unlock();
}

void SeaBee3GUIIce::setSalientPointsEnabled(bool enabled)
{
  itsUpdateMutex.lock();
  itsSalientPointsEnabled = enabled;
  itsUpdateMutex.unlock();
}

// ######################################################################
void SeaBee3GUIIce::updateMessage(const RobotSimEvents::EventMessagePtr& eMsg,
                                  const Ice::Current&)
{
  itsUpdateMutex.lock();
  //Get a retina message
  if(eMsg->ice_isA("::RobotSimEvents::RetinaMessage"))
    {
      RobotSimEvents::RetinaMessagePtr msg = RobotSimEvents::RetinaMessagePtr::dynamicCast(eMsg);
      if(Ice2Image<PixRGB<byte> >(msg->img).initialized())
        {
          Image<PixRGB<byte> > retinaImage = Ice2Image<PixRGB<byte> >(msg->img);

          if(msg->cameraID == "FwdCamera")
            {
              if(itsFwdRetinaImages.size() == MSG_BUFF_MAX)
                {
                  LINFO("Dropping fwd retina msg");
                  itsFwdRetinaImages.pop_front();
                }

              itsFwdRetinaImages.push_back(retinaImage);
              itsFwdRetinaMsgCounter++;
            }
          else if(msg->cameraID == "DwnCamera")
            {
              if(itsDwnRetinaImages.size() == MSG_BUFF_MAX)
                itsDwnRetinaImages.pop_front();

              itsDwnRetinaImages.push_back(retinaImage);
              itsDwnRetinaMsgCounter++;
            }
          else if(msg->cameraID == "BuoyColorSegmenter" && itsOrangeSegEnabled)
            {
              if(itsOrangeSegImages.size() == MSG_BUFF_MAX)
                {
                  itsOrangeSegImages.pop_front();
                  LINFO("Dropping buoy color segmenter msg");
                }

              itsOrangeSegImages.push_back(retinaImage);
              itsVisionMsgCounter++;
            }
        }
    }
  else if(eMsg->ice_isA("::RobotSimEvents::BeeStemMessage"))
    {

      RobotSimEvents::BeeStemMessagePtr msg = RobotSimEvents::BeeStemMessagePtr::dynamicCast(eMsg);

      BeeStemData d;
      d.heading = msg->compassHeading;
      d.externalPressure = msg->externalPressure;
      d.internalPressure = msg->internalPressure;
      d.headingPIDOutput = msg->headingOutput;
      d.depthPIDOutput = msg->depthOutput;
      d.desiredDepth = msg->desiredDepth;
      d.killSwitch = (msg->killSwitch == 1) ? true : false;

      if(itsBeeStemData.size() == MSG_BUFF_MAX)
        {
          LINFO("Dropping bee stem msg");
          itsBeeStemData.pop_front();
        }

      itsBeeStemData.push_back(d);
      itsBeeStemMsgCounter++;
    }
  else if(eMsg->ice_isA("::RobotSimEvents::SalientPointMessage"))
    {
      RobotSimEvents::SalientPointMessagePtr msg = RobotSimEvents::SalientPointMessagePtr::dynamicCast(eMsg);
      if(itsSalientPoints.size() == MSG_BUFF_MAX)
        itsSalientPoints.pop_front();

      itsSalientPoints.push_back(Point2D<int>((int)(itsFwdVisionImage.getWidth()*msg->x),
                                              (int)(itsFwdVisionImage.getHeight()*msg->y)));
      itsVisionMsgCounter++;
    }

  itsUpdateMutex.unlock();
}
//     itsGUIForm->setThrusterMeters(
//         0,
//         (int)msg->thruster1,
//         (int)msg->thruster2,
//         (int)msg->thruster3,
//         (int)msg->thruster4,
//         (int)msg->thruster5,
//         (int)msg->thruster6,
//         0
//         );

//   // Get a XBox360RemoteControl Message
//   else if(eMsg->ice_isA("::RobotSimEvents::JoyStickControlMessage"))
//   {
//     RobotSimEvents::JoyStickControlMessagePtr msg = RobotSimEvents::JoyStickControlMessagePtr::dynamicCast(eMsg);

//     itsGUIFormMutex.lock();
//     if(itsGUIRegistered)
//     {
//       if(msg->axis >= 0)
//         itsGUIForm->setJSAxis(msg->axis, msg->axisVal);
//       else
//         LINFO("Button[%d] = %d",msg->button, msg->butVal);
//     }
//     itsGUIFormMutex.unlock();
//   }
//   // Get a BeeStem Message

//   else if(eMsg->ice_isA("::RobotSimEvents::VisionRectangleMessage"))
//     {
//       RobotSimEvents::VisionRectangleMessagePtr msg = RobotSimEvents::VisionRectangleMessagePtr::dynamicCast(eMsg);
//       itsGUIFormMutex.lock();
//       if(itsGUIRegistered)
//         {
//           if(msg->isFwdCamera)
//             {
//               for(uint i = 0; i < msg->quads.size(); i++)
//                 {
//                   itsGUIForm->pushFwdRectangle(msg->quads[i]);
//                 }
//             }
//           else
//             {
//               for(uint i = 0; i < msg->quads.size(); i++)
//                 {

//                   itsGUIForm->pushRectRatio(msg->quads[i].ratio);
//                   itsGUIForm->pushRectAngle(msg->quads[i].angle);

//                   itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].tl.i,msg->quads[i].tl.j));
//                   itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].tr.i, msg->quads[i].tr.j));
//                   itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].br.i, msg->quads[i].br.j));
//                   itsGUIForm->pushContourPoint(Point2D<int>(msg->quads[i].bl.i, msg->quads[i].bl.j));
//                 }
//             }
//         }
//       itsGUIFormMutex.unlock();
//     }
//   else if(eMsg->ice_isA("::RobotSimEvents::SalientPointMessage"))
//     {
//       RobotSimEvents::SalientPointMessagePtr msg = RobotSimEvents::SalientPointMessagePtr::dynamicCast(eMsg);
//       itsGUIFormMutex.lock();
//       if(itsGUIRegistered)
//         {
//           itsGUIForm->setSalientPoint(Point2D<float>(msg->x,msg->y));
//         }
//       itsGUIFormMutex.unlock();
//     }
//   else if(eMsg->ice_isA("::RobotSimEvents::MovementControllerMessage"))
//     {
//       RobotSimEvents::MovementControllerMessagePtr msg = RobotSimEvents::MovementControllerMessagePtr::dynamicCast(eMsg);
//       itsGUIFormMutex.lock();
//       if(itsGUIRegistered)
//         {
//           itsGUIForm->setSensorVotes(msg->votes);
//           itsGUIForm->setCompositeHeading(msg->compositeHeading);
//           itsGUIForm->setCompositeDepth(msg->compositeDepth);
//         }
//       itsGUIFormMutex.unlock();
//     }

// ######################################################################
void SeaBee3GUIIce::registerGUI(SeaBee3MainDisplayForm* form)
{
  itsGUIFormMutex.lock();
  itsGUIForm = form;
  itsGUIRegistered = true;
  itsGUIFormMutex.unlock();
}
#endif

