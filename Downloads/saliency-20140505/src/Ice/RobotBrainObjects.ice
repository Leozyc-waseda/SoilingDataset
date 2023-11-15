#include <Ice/ImageIce.ice> 
#include <Ice/RobotSimEvents.ice> 

#ifndef ROBOTBRAIN_OBJECTS
#define ROBOTBRAIN_OBJECTS

module RobotBrainObjects {

  const int RobotBrainPort            = 20004;
  const int SaliencyModulePort        = 20005;
  const int RetinaPort                = 20006;
  const int SeaBeeGUIPort             = 20007;

  interface Retina extends RobotSimEvents::Events {
  };

  interface SaliencyModule extends RobotSimEvents::Events {
  };

  //The PMC revices action commands and acts on them
  interface PrimaryMotorCortex extends RobotSimEvents::Events {
  };

  interface PrimarySomatosensoryCortex extends RobotSimEvents::Events {
  };

  interface LateralGeniculateNucleus extends RobotSimEvents::Events {
  };

  interface Hippocampus extends RobotSimEvents::Events {
  };

  interface SupplementaryMotorArea extends RobotSimEvents::Events {
  };

  interface PrefrontalCortex extends RobotSimEvents::Events {
  };

  interface InferotemporalCortex extends RobotSimEvents::Events {
  };

  interface Observer extends RobotSimEvents::Events {
  };

};

#endif

