#ifndef SeaBee3GUICommunicator_H
#define SeaBee3GUICommunicator_H

#include "Qt/ui/SeaBee3GUI.h"
#include "Robots/RobotBrain/RobotBrainComponent.H"

class SeaBee3MainDisplayForm;

class SeaBee3GUICommunicator : public RobotBrainComponent
{
public:
  SeaBee3GUICommunicator(OptionManager& mgr,
                         const std::string& descrName = "SeaBee3Communicator",
                         const std::string& tagName = "SeaBee3Communicator");

  virtual void updateMessage(const RobotSimEvents::EventMessagePtr& eMsg,
                             const Ice::Current&);

  virtual void registerTopics();

  virtual void evolve();

  void registerGUI(SeaBee3MainDisplayForm* form);

  void toggleCamera(std::string id, bool active);

  void updatePoseSettings(int updateSelect, int heading, int depth, int speed);

  void updatePID(int pidSelect, float k, float p, float i, float d);

  void enablePID();

  void disablePID();

        void SeaBeeInjectorMsg(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j);

private:
  SeaBee3MainDisplayForm *itsGUIForm;

  bool itsGUIRegistered;
  IceUtil::Mutex itsGUIFormMutex;

  Image<PixRGB<byte> > itsFwdImg;
  Image<PixRGB<byte> > itsDwnImg;
  IceUtil::Mutex itsImageMutex;

  int itsFwdFrameCount;
  int itsLastFwdFrameCount;
  int itsDwnFrameCount;
  int itsLastDwnFrameCount;

};

#endif

