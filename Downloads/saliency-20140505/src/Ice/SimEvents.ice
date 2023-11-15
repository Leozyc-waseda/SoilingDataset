#include <Ice/ImageIce.ice> 

#ifndef SIM_EVENTS
#define SIM_EVENTS

module SimEvents {

  class EventMessage {
  };

  interface Events {
    void evolve (EventMessage eMsg);
  };

  //SimEvent messages
  class RetinaMessage extends EventMessage {
    ImageIceMod::ImageIce img;
  };

  class VisualCortexMessage extends EventMessage {
    ImageIceMod::ImageIce vco;
  };

  //Struct for saliency map info
  struct LocInfo
  {
    ImageIceMod::Point2DIce lowresMaxpos;
    ImageIceMod::Point2DIce fullresMaxpos;
    byte maxval;
  };
  sequence<LocInfo> LocInfoSeq;

  class SaliencyMapMessage extends EventMessage {
    //the current saliency map
    ImageIceMod::ImageIce smap;
    //the top N most salient locations
    LocInfoSeq nMostSalientLoc;
  };

  class SaliencyMapBiasMessage extends EventMessage {
    //the bias map
    ImageIceMod::ImageIce biasImg;
  };

  struct TrackInfo
  {
    ImageIceMod::Point2DIce pos;
    float err;
  };
  sequence<TrackInfo> TrackInfoSeq;

  class VisualTrackerMessage extends EventMessage {
    //the locations currently being tracked 
    TrackInfoSeq trackLocs;
  };

  class VisualTrackerBiasMessage extends EventMessage {
    //the locations to track
    TrackInfoSeq locToTrack;
  };

  //struct for CameraCtrl messages
  class CameraCtrlBiasMessage extends EventMessage {
    bool zoomIn;
    bool zoomOut;
    bool initPtz;
  };

  class CameraCtrlMessage extends EventMessage {
    bool zoomDone;
    bool initPtzDone;
    int pan;
    int tilt;
  };



  //struct for segment info
  struct SegInfo
  {
    ImageIceMod::Point2DIce loc;
    ImageIceMod::RectangleIce rect;
    ImageIceMod::ImageIce img;
  };

  sequence<SegInfo>  SegInfoSeq;

  class SegmenterMessage extends EventMessage {
    //get the seqgmeneted region
    SegInfoSeq segLocs;
  };

  class SegmenterBiasMessage extends EventMessage {
    //set the locations to segment
    SegInfoSeq regionsToSeg;
  };

  class InfrotemporalCortexBiasMessage extends EventMessage {
    ImageIceMod::ImageIce img;
  };

  class InfrotemporalCortexMessage extends EventMessage {
    string objectName;
    ImageIceMod::Point2DIce objLoc;
    float rotation;
    float confidence;
  };

  class HippocampusBiasMessage extends EventMessage {
    ImageIceMod::Point2DIce loc;
    int pan;
    int tilt;
    float rotation;
    string objectName;
    ImageIceMod::ImageIce img;
  };

  struct ObjectState
  {
    string objectName;
    string objectMsg;
    bool inDanger;
    ImageIceMod::Point2DIce loc;
    ImageIceMod::Point3DIce pos;
    float rotation;
    float size;
  };

  sequence<ObjectState>  ObjectsStateSeq;

  class HippocampusMessage extends EventMessage {
    ObjectsStateSeq objectsState;
  };

  struct ArmPosition
  {
    float x;
    float y;
    float z;
    float rot;
    float roll;
    float gripper;
  };

  class PrimaryMotorCortexBiasMessage extends EventMessage {
    ArmPosition armPos;
    bool moveArm;
  };

  class PrimaryMotorCortexRelativeServoMessage extends EventMessage {
     int axis;
     int change;
  };

  class PrimaryMotorCortexResetEncodersMessage extends EventMessage {};

  class PrimaryMotorCortexMessage extends EventMessage {
    bool targetReached;
  };

  class GUIInputMessage extends EventMessage {
    string msg;
    bool rollDice;
  };

  class GUIOutputMessage extends EventMessage {
    int key;
    ImageIceMod::Point2DIce loc;
    int dice;
  };


};

#endif
