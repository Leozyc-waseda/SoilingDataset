

#include <Ice/ImageIce.ice> 
#include <Ice/SimEvents.ice> 

#ifndef BRAIN_OBJECTS
#define BRAIN_OBJECTS

module BrainObjects {

  const int RetinaPort = 20000;
  const int VisualCortexPort = 20001;
  const int SaliencyMapPort = 20002;
  const int SegmenterPort = 20003;
  const int VisualTrackerPort = 20004;
  const int InferoTemporalPort = 20005;
  const int PrefrontalCortexPort = 20006;
  const int SimulationViewerPort = 20007;
  const int PTZPort = 20008;
  const int HippocampusPort = 20009;
  const int PrimaryMotorCortexPort = 20010;
  const int HomeInterfacePort = 20011;

  //The posts a RetinaMessage
  interface Retina extends SimEvents::Events {
    ImageIceMod::ImageIce getOutput();
  };

  //The Visual Cortex gets a Retina Message and posts a saliency map 
  interface VisualCortex extends SimEvents::Events {
  };

  //The Saliency Map gets a VisualCortex Message and posts the top most salient locations
  interface SaliencyMap extends SimEvents::Events {
  };

  //The Tracker message excepts a RetinaImage and LocInfo seq to track
  //It posts the current track locations
  interface VisualTracker extends SimEvents::Events {
  };

  //The Pan tilt and zoom the camera
  interface PTZ extends SimEvents::Events {
  };

  //The segmenter message excepts a RetinaImage and LocInfo seq to segment
  //It posts the current segments 
  interface Segmenter extends SimEvents::Events {
  };

  //The IT does object recognition
  interface InferotemporalCortex extends SimEvents::Events {
  };

  //The Hippocampus has state of objects
  interface Hippocampus extends SimEvents::Events {
  };

  //The pmc
  interface PrimaryMotorCortex extends SimEvents::Events {
  };

  //The PC hold the state machine to make decisions about various actions
  interface PrefrontalCortex extends SimEvents::Events {
  };

  //The SimViewer show the various state of all modules
  interface SimulationViewer extends SimEvents::Events {
  };
  
  interface HomeInterface extends SimEvents::Events {
  };

};

#endif

