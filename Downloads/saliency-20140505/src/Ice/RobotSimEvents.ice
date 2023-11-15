#include <Ice/ImageIce.ice> 

#ifndef ROBOTSIM_EVENTS
#define ROBOTSIM_EVENTS

module RobotSimEvents {

  class EventMessage {
  };

  interface Events {
    void updateMessage (EventMessage eMsg);
  };

  //Action messages
  class ActionMessage extends EventMessage {
    float transVel;
    float rotVel;
  };
  
  class GPSMessage extends EventMessage {
    float xPos;
    float yPos;
    float orientation;
  };

  class MotionMessage extends EventMessage {
    float distance;
    float angle;
  };

  class RetinaMessage extends EventMessage {
    ImageIceMod::ImageIce img;
    string cameraID;
  };

  sequence<ImageIceMod::SensorVote> SensorVoteVector;
  class MovementControllerMessage extends EventMessage {
    SensorVoteVector votes;
    float compositeHeading;
    float compositeDepth;
  };

  class JoyStickControlMessage extends EventMessage {
    string axisName;
    int axis;
    int axisVal;
    int button;
    int butVal;
  };

  class BeeStemConfigMessage extends EventMessage {
  	int deviceToFire;
    int desiredHeading;
    int desiredDepth;
    int desiredSpeed;
    int updateDesiredValue;

    float headingK;
    float headingP;
    float headingI;
    float headingD;
    int updateHeadingPID;   

    float depthK;
    float depthP;
    float depthI;
    float depthD;
    int updateDepthPID;

    int enablePID;
    int enableVal;
  };
  
  sequence<float> FloatVector;
  
  class IMUDataServerMessage extends EventMessage {
  	double temp;
  	ImageIceMod::Point3DIce accel;
  	ImageIceMod::Point3DIce gyro;
  	ImageIceMod::Point3DIce mag;
  	int angleMode; //0 = quaternian; 1 = euler; 2 = cosine matrix
  	FloatVector orientation;
  };
  
  sequence<int> IntVector;
  
  class BeeStemMotorControllerMessage extends EventMessage {
  	IntVector mask;
  	IntVector values;
  };
  	
  class LocalizationMessage extends EventMessage {
  	ImageIceMod::Point2DIce pos;
  	float heading;
  	int mode;
  	ImageIceMod::WaypointIce currentWaypoint;
  };

  //Note: Please annotate the fields of this message
  class BeeStemMessage extends EventMessage {
    int accelX;
    int accelY;
    int accelZ;
    int compassHeading;
    int compassPitch;
    int compassRoll;
    int internalPressure;
    int externalPressure;
    int desiredHeading;
    int desiredDepth;
    int desiredSpeed;	
    int headingK;
    int headingP;
    int headingD;
    int headingI;
    int headingOutput; // output value of heading PID loop
    int depthK;
    int depthP;
    int depthD;
    int depthI;
    int depthOutput; // output value of depth PID loop
    int thruster1;
    int thruster2;
    int thruster3;
    int thruster4;
    int thruster5;
    int thruster6;
    int killSwitch;
  };

  class ChatMessage extends EventMessage {
    string text;
    string from;
  };
 
  class RemoteControlMessage extends EventMessage {
    int x;
    int y;
    int imgStreamId;
  };

  class StateMessage extends EventMessage {
    float xPos;
    float yPos;
    float orientation;
  };

  class GoalStateMessage extends EventMessage {
    float xPos;
    float yPos;
    float orientation;
  };

  class GoalProgressMessage extends EventMessage {
    float err;
  };


  class AttendedRegionMessage extends EventMessage {
    ImageIceMod::ImageIce img;

    //The following variables are applicable only for object recognition training.
    //Ideally, these should constitute a different message type
    int objId;
    string name;
    float objWidth;
    float objHeight;

    ImageIceMod::Point2DIce attTopLeft; //The top left coordinate of a selected region 
    int    attWidth;    //The dimensions of a selected region 
    int    attHeight;    //The dimensions of a selected region 
  };

  class ObjectMessage extends EventMessage {
    int id; //the object id
    string name; //the name of the object
    float score; //the matching score
    int nKeyp; //the number of keypoints
    float dist; //actual distance between keypoints

    ImageIceMod::Point2DIce tl; //the matched outline
    ImageIceMod::Point2DIce tr; //the matched outline
    ImageIceMod::Point2DIce br; //the matched outline
    ImageIceMod::Point2DIce bl; //the matched outline
  };

	class SeaBeePositionMessage extends EventMessage {
		float x;
		float y;
		float orientation;
		float xVar;
		float yVar;
	};
	
  class SeaBeeStateConditionMessage extends EventMessage {
		int StartOver;
		int InitDone;
		int GateFound;
		int GateDone;
		int ContourFoundFlare;
		int FlareDone;
		int ContourFoundBarbwire;
		int BarbwireDone;
		int ContourFoundBoxes;
		int BombingRunDone;
		int BriefcaseFound;
		int TimeUp;
		int PathFollowFinished;
	};


  sequence<ImageIceMod::QuadrilateralIce> QuadrilateralIceVector;
  class VisionRectangleMessage extends EventMessage {
    //identify quadrilaterals in the field of vision and send coords
    QuadrilateralIceVector quads;
    bool isFwdCamera; // whether or not recognizd  with forward camera 
  };

  class StraightEdgeMessage extends EventMessage {
    //identify lines in the field of vision and send coords
    ImageIceMod::LineIce line;
    bool isFwdCamera; // whether or not recognizd  with forward camera 
  };

  enum SeaBeeObjectType { PLATFORM, GATE, PIPE, FLARE, BIN, BARBWIRE, MACHINEGUN, OCTAGON };
  class VisionObjectMessage extends EventMessage {
    SeaBeeObjectType objectType;         //Make sure to sync up with the rest of the team (especially the mapping people) 
                               //to ensure that these strings are all the same. Alternatively, make this an enum
                               //and use it everywhere (GUI, Mapping, etc...)

    bool forwardCamera;        //Whether or not this was sighted in the forward camera

    ImageIceMod::Point3DIce objectPosition; //The extrapolated 3D position of the object
                               //X is left - right on the submarine
                               //Y is up and down on the submarine
                               //Z is forwards and backwards on the submarine

    ImageIceMod::Point3DIce objectVariance; //The uncertainty in the object position
  };


  sequence<byte> FeatureVector;
  struct LandmarkInfo 
  {
    int id;
    string name;
    FeatureVector fv;
    float range;
    float bearing;
    float prob;
  };

  sequence<LandmarkInfo> LandmarksInfo;

  class LandmarksMessage extends EventMessage {
    LandmarksInfo landmarks;
  };

  class CameraConfigMessage extends EventMessage {
    string cameraID;
    bool active;
  };

  class SalientPointMessage extends EventMessage {
    float x;
    float y; 
    int saliency; 
    string imageDescr;           //A description of the image on which the saliency computation was performed
                                // eg. the camera descriptor
  };

  class BuoyColorSegmentMessage extends EventMessage {
    float x; //floats from 0 to 1 indicating relative location on camera
    float y;

    float size;
  };

  class PipeColorSegmentMessage extends EventMessage {
    float x; //floats from 0 to 1 indicating relative location on camera
    float y;

    float size;
  };

  class BuoyColorSegmentConfigMessage extends EventMessage {
    //sets threshold values for Buoy color segmenter
    double HUT; //Hue upper and lower thresholds
    double HLT;
    double SUT; //Saturation
    double SLT;
    double VUT; //Value
    double VLT;
  };
  
  class PipeColorSegmentConfigMessage extends EventMessage {
    //sets threshold values for pipe color segmenter
    double HUT; //Hue upper and lower thresholds
    double HLT;
    double SUT; //Saturation
    double SLT;
    double VUT; //Value
    double VLT;
  };
  
};

#endif
