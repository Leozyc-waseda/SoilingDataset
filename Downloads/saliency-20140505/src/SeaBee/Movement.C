#ifndef __MOVEMENT_AGENT_C__
#define __MOVEMENT_AGENT_C__

#include "Movement.H"


// ######################################################################
MovementAgent::MovementAgent(OptionManager& mgr,
                             nub::soft_ref<AgentManager> ama,
                             nub::soft_ref<SubController> subController,
                             const std::string& name) :
  SubmarineAgent(mgr, ama, name),
  itsDepthErrThresh("setDepthErrThresh", this, 0, ALLOW_ONLINE_CHANGES),
  itsHeadingErrThresh("setHeadingErrThresh", this, 0, ALLOW_ONLINE_CHANGES),
  pipeP("pipeP", this, 0, ALLOW_ONLINE_CHANGES),
  pipeI("pipeI", this, 0, ALLOW_ONLINE_CHANGES),
  pipeD("pipeD", this, 0, ALLOW_ONLINE_CHANGES),
  itsPipePID(pipeP.getVal(), pipeI.getVal(), pipeD.getVal(), -20, 20, 5, 0, 0, 100, -100, 150, true, 1, 25, -25),
  setDiveValue("setDiveValue", this, 0, ALLOW_ONLINE_CHANGES),
  setGoStraightTimeValue("setGoStraightTimeValue", this, 0, ALLOW_ONLINE_CHANGES),
  setSpeedValue("setSpeedValue", this, 20, ALLOW_ONLINE_CHANGES),
  setHeadingValue("setHeadingValue", this, 0, ALLOW_ONLINE_CHANGES),
  setRelative("setRelative", this, false, ALLOW_ONLINE_CHANGES),
  setTimeout("setTimeout", this, 100, ALLOW_ONLINE_CHANGES),
  itsSubController(subController),
{
  itsFrameNumber = 0;
  itsTimer.reset(new Timer(1000000));
}


// ######################################################################
MovementAgent::~MovementAgent()
{

}

// ######################################################################
//Scheduler
bool MovementAgent::pickAndExecuteAnAction()
{
  return true;
}

// ######################################################################
bool MovementAgent::dive(int depth, bool relative, int timeout)
{
  if(relative)
    {
      itsSubController->setDepth(depth + itsSubController->getDepth());
    }
  else
    {
      itsSubController->setDepth(depth);
    }

  int time = 0;

  while(itsSubController->getDepthErr() > itsDepthErrThresh.getVal())
    {
      usleep(10000);
      if(time++ > timeout) return false;
      //<TODO mmontalbo> turn off diving
    }

  return true;
}

// ######################################################################
bool MovementAgent::goStraight(int speed, int time)
{
  itsSubController->setSpeed(speed);
  sleep(time);
  itsSubController->setSpeed(0);
  return true;
}

// ######################################################################
bool MovementAgent::setHeading(int heading, bool relative, int timeout)
{
  if(relative)
    {
      itsSubController->setHeading(heading + itsSubController->getHeading());
    }
  else
    {
      itsSubController->setHeading(heading);
    }

  int time = 0;

  while(itsSubController->getHeadingErr() > itsHeadingErrThresh.getVal())
    {
      usleep(10000);
      if(time++ > timeout) return false;
      //<TODO mmontalbo> turn off turning
    }

  return true;
}

// ######################################################################
int MovementAgent::trackPipe(const Point2D<int>& pointToTrack,
                                  const Point2D<int>& desiredPoint)
{
  float pipeCorrection = (float)itsPipePID.update(pointToTrack.i, desiredPoint.i);

  //  itsSubController->setHeading(itsSubController->getHeading()
  //                          + pipeCorrection);

  itsSubController->setTurningSpeed(pipeCorrection);

  return abs(pointToTrack.i - desiredPoint.i);
}

// ######################################################################
void MovementAgent::paramChanged(ModelParamBase* const param,
                                 const bool valueChanged,
                                 ParamClient::ChangeStatus* status)
{
  if (param == &setDiveValue && valueChanged)
    {
      if(dive(setDiveValue.getVal(), setRelative.getVal(), setTimeout.getVal()))
        LINFO("Dive completed successfully");
      else
        LINFO("Dive failed");
    }
  else if(param == &setGoStraightTimeValue && valueChanged)
    {
      if(goStraight(setSpeedValue.getVal(), setGoStraightTimeValue.getVal()))
        LINFO("Go straight completed successfully");
      else
        LINFO("Go straight failed");
    }
  else if(param == &setHeadingValue && valueChanged)
    {
      if(setHeading(setHeadingValue.getVal(), setRelative.getVal(), setTimeout.getVal()))
        LINFO("Turn completed successfully");
      else
        LINFO("Turn failed");
    }
  //////// Pipe PID constants/gain change ////////
  else if (param == &pipeP && valueChanged)
    itsPipePID.setPIDPgain(pipeP.getVal());
  else if(param == &pipeI && valueChanged)
    itsPipePID.setPIDIgain(pipeI.getVal());
  else if(param == &pipeD && valueChanged)
    itsPipePID.setPIDDgain(pipeD.getVal());

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
