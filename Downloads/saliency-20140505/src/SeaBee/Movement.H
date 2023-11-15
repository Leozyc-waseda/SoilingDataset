#ifndef __MOVEMENT_AGENT_H__
#define __MOVEMENT_AGENT_H__

#include "Captain.H"
#include "AgentManager.H"
#include "SubController.H"
#include "SubmarineAgent.H"
#include "Point3D.H"

class AgentManager;
class CaptainAgent;

class MovementAgent : public SubmarineAgent
{

public:

  MovementAgent(OptionManager& mgr,
                nub::soft_ref<AgentManager> ama,
                nub::soft_ref<SubController> subController,
                const std::string& name = "MovementAgent");


  ~MovementAgent();

  inline void setCaptainAgent
  (nub::soft_ref<CaptainAgent> pfc);

private:

  bool pickAndExecuteAnAction();


  bool dive(int depth, bool relative = false, int timeout = 0);

  bool goStraight(int speed, int time);

  bool setHeading(int heading, bool relative = false, int timeout = 0);

  int trackPipe(const Point2D<int>& pointToTrack, const Point2D<int>& desiredPoint);

  void paramChanged(ModelParamBase* const param,
                    const bool valueChanged,
                    ParamClient::ChangeStatus* status);

  NModelParam<int>  itsDepthErrThresh;
  NModelParam<int>  itsHeadingErrThresh;

  NModelParam<float> pipeP;
  NModelParam<float> pipeI;
  NModelParam<float> pipeD;
  PID<float> itsPipePID;

  NModelParam<int>  setDiveValue;
  NModelParam<int>  setGoStraightTimeValue;
  NModelParam<int>  setSpeedValue;
  NModelParam<int>  setHeadingValue;
  NModelParam<bool>  setRelative;
  NModelParam<int>  setTimeout;

  nub::soft_ref<SubController> itsSubController;
  nub::soft_ref<CaptainAgent> itsCaptainAgent;
  nub::soft_ref<AgentManager> itsAgentManager;

  uint itsFrameNumber;

  rutz::shared_ptr<Timer> itsTimer;

};


// ######################################################################
inline void MovementAgent::setCaptainAgent
(nub::soft_ref<CaptainAgent> ca)
{
  itsCaptainAgent = ca;
}

#endif
