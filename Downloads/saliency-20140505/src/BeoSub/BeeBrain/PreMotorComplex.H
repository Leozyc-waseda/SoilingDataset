#ifndef __PREMOTOR_COMPLEX_H__
#define __PREMOTOR_COMPLEX_H__

#include "Component/ModelManager.H"
#include "BeoSub/BeeBrain/PreFrontalCortex.H"
#include "BeoSub/BeeBrain/ActionInput.H"
#include "BeoSub/BeeBrain/ComplexMovement.H"
#include "BeoSub/BeeBrain/AgentManagerA.H"
#include "SeaBee/SubController.H"

#define INVALID -1.0

class AgentManagerA;
class PreFrontalCortexAgent;


class PreMotorComplex {

public:

  PreMotorComplex(nub::soft_ref<SubController> motor, /*rutz::shared_ptr<PreFrontalCortexAgent> pfc,*/ std::string name);
  PreMotorComplex(std::string name);
  ~PreMotorComplex();

  inline void setPreFrontalCortexAgent
  (rutz::shared_ptr<PreFrontalCortexAgent> pfc);


  //! starts the PreMotorComplex
  void start();
  //! stops everything
  inline void stop() { itsStop = itsInterrupt = true; }


  //! stop all movements. stay at current depth and heading, using PID
  inline void msgHalt() { itsInterrupt = true; }
  //! run ComplexMovement a, if there is a ComplexMovement already running,
  //! will cause interrupt and start new movement
  void run(rutz::shared_ptr<ComplexMovement> a);

  inline void test() { LINFO("you called me"); }



  // world/sensor moves
  // turn to a goal absolute angle (heading)
  // NOTE: will this be in radians?
  void turn(const SensorInput goal);
  // move forward/reverse in reference to goal distance (meters)
  // forward: positive, reverse: negative
  void forward(const SensorInput goal);
  // dive/surface to a goal depth in reference to goal depth (meters)
  // dive: positive, surface: negative
  void dive(const SensorInput goal);
  // types of searches
  void sweep(const SensorInput goal);


  // vision moves
  // turn to a goal pixel angle, or turn to a goal pixel point
  void vis_turn(const VisionInput& goal);
  // move forward/reverse to a goal pixel point
  void vis_forward(const VisionInput& goal);
  // dive/surface to a goal pixel point
  void vis_dive(const VisionInput& goal);
  // center on a pixel point
  void vis_center(const VisionInput& goal);
  // lost image retract search
  void vis_lost(const VisionInput& goal);





private:


  void run();



//   Angle itsHeading;
//   Angle itsPitch;
//   Angle itsRoll;
//   float itsTemp;
//   float itsInternalPressure;
//   float itsDepth;

//   //possibly its position relative to the pinger;
//   Point3D itsAttitude;

  rutz::shared_ptr<ComplexMovement> itsCurrentMove;
  rutz::shared_ptr<ComplexMovement> itsNextMove;

  nub::soft_ref<SubController> itsPrimitiveMotor;
  rutz::shared_ptr<PreFrontalCortexAgent> itsCortex;

  bool itsInterrupt;
  bool itsStop;
  std::string itsName;

};

// ######################################################################
inline void PreMotorComplex::setPreFrontalCortexAgent
(rutz::shared_ptr<PreFrontalCortexAgent> pfc)
{
  itsCortex = pfc;
}

#endif
