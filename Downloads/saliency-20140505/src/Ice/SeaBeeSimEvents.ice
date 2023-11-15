#include <Ice/ImageIce.ice>
#include <Ice/RobotSimEvents.ice>

#ifndef ROBOTSIM_EVENTS
#define ROBOTSIM_EVENTS

module SeaBeeSimEvents
{
  struct Particle
  {
     float x;
     float y;
     float p;
  };

  sequence<Particle> ParticleList;
  
  class ParticleMessage extends RobotSimEvents::EventMessage
  {
     ParticleList particles;
  };
};

#endif


