#ifndef SeaBee3GUIIce_H
#define SeaBee3GUIIce_H

#include "Qt/ui/SeaBee3GUI2.h"
#include "Robots/RobotBrain/RobotBrainComponent.H"
#include "Util/Timer.H"
#include "GUI/CompassMeter.H"
#include "GUI/DepthMeter.H"
#include "GUI/PressureMeter.H"
#include "GUI/CircleFillMeter.H"

class SeaBee3MainDisplayForm;

class BeeStemData
{
public:
  float heading;
  float externalPressure;
  float internalPressure;
  int headingPIDOutput;
  int depthPIDOutput;
  int desiredDepth;
  bool killSwitch;
};

class SeaBee3GUIIce : public RobotBrainComponent
{
public:
  SeaBee3GUIIce(OptionManager& mgr,
                const std::string& descrName = "SeaBee3GUIIce",
                const std::string& tagName = "SeaBee3GUIIce");

  virtual void updateMessage(const RobotSimEvents::EventMessagePtr& eMsg,
                             const Ice::Current&);

  virtual void registerTopics();

  virtual void evolve();

  void registerGUI(SeaBee3MainDisplayForm* form);

  void setOrangeSegEnabled(bool enabled);
  void setSalientPointsEnabled(bool enabled);

private:
  void updateFwdImg();

  void updateDwnImg();

  void updateBeeStemData();

  void updateSaliencyImage();


  SeaBee3MainDisplayForm *itsGUIForm;

  bool itsGUIRegistered;
  IceUtil::Mutex itsGUIFormMutex;

  IceUtil::Mutex itsUpdateMutex; // Locks lists for incoming updates

  std::list<Image<PixRGB<byte> > > itsFwdRetinaImages;
  char itsFwdRetinaMsgCounter;

  std::list<Image<PixRGB<byte> > > itsDwnRetinaImages;
  char itsDwnRetinaMsgCounter;

  std::list<Image<PixRGB<byte> > > itsOrangeSegImages;
  bool itsOrangeSegEnabled;

  std::list<Image<PixRGB<byte> > > itsRedSegImages;
  char itsRedSegImagesSize;

  std::list<Point2D<int> > itsSalientPoints;
  int itsSalientPointsSize;
  bool itsSalientPointsEnabled;

  char itsVisionMsgCounter;

  std::list<BeeStemData> itsBeeStemData;
  char itsBeeStemMsgCounter;
  CompassMeter itsCompassMeter;
  DepthMeter itsDepthMeter;
  PressureMeter itsPressureMeter;
  CircleFillMeter itsCircleFillMeter;
  TargetLineMeter itsTargetLineMeter;

  Timer itsTimer;

  Image<PixRGB<byte> > itsFwdVisionImage;
  Image<PixRGB<byte> > itsDwnVisionImage;

};

#endif

