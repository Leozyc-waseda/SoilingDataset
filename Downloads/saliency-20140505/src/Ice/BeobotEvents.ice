#include <Ice/ImageIce.ice> 
#include <Ice/RobotSimEvents.ice>
 
#ifndef BEOBOT_EVENTS
#define BEOBOT_EVENTS

module BeobotEvents 
{
  sequence<double> DoubleSeq;  
  
  /////////////////////////////////////////////////////////////////////////////////////////
  // Hardware Related Messages
  /////////////////////////////////////////////////////////////////////////////////////////
	
  // Message for log folder name
  class LogFolderNameMessage extends RobotSimEvents::EventMessage
  {
    int RequestID;
    string logFolderName;
  };

  // request for log folder name
  class LogFolderNameRequest extends RobotSimEvents::EventMessage
  {
    int    RequestID;
  };
  // Tell all the program to reset itself 
	// resetID can be use for reset specific app or every one
	// resetID : 0 reset all
	// resetID : 1 reset BeoPilot
	// resetID : 2 reset BeoNavigator
	// resetID : 3 reset BeoLocalizer
	// resetID : 4 reset BeoRoadFinder
  class ResetRequest extends RobotSimEvents::EventMessage 
  {
		int RequestAppID;
    int ResetID ;
		
  };

  // MotorMessage: Sent from BeoPilot to keep the system up to date on the 
  //              current motor speeds
  class MotorMessage extends RobotSimEvents::EventMessage
  {
    // mode 0: manual; 1: semi-auto; 2: auto  
    int rcMode;

    int motor1;//command send to motor driver
    int motor2;//command sned to motor driver

    double transVel;//computer requested trans speed
    double rotVel;//computer requested rot speed

    double encoderX;//dx
    double encoderY;//dx
    double encoderOri;  // current heading (not d_theta)  

    double rawEncoderX;//dx
    double rawEncoderY;//dx
    double rawEncoderOri;  // current heading (not d_theta)  

    double rcTransVel;
    double rcTransCap;
    double rcRotVel;
    double rcRotCap;

    double robotTransVel;// current robot moving speed,measured from encoder,in meter/sec
    double robotRotVel;  // current robot speed,measured from encoder/imu, in rad/sec

    double imuHeading;//current imu heading
    double trajLen;//totol trajectory length
    double dist;//total straight distance from origin

    double battery;//battery voltage
    int RequestID;
  };

  // MotorRequest: Ask the BeoPilot module to set a new speed for the robot
  class MotorRequest extends RobotSimEvents::EventMessage
  {
    int RequestAppID;
    double transVel;
    double rotVel;
  };

  // MotorRequest: 
  class CornerLocationMessage extends RobotSimEvents::EventMessage
  {
    int cornerLocation;
  };

  // SonarMessage: Sent from BeoSonar to inform the system of new sonar readings
  class SonarMessage extends RobotSimEvents::EventMessage
  {
    // The distances from each ray in millimeters
    DoubleSeq distances;

    // The angle of each sonar unit. These angles should not change from message
    // to message, but are here just for clarity.
    DoubleSeq angles;
  };

  // GpsMessage: Sent from GPS in the sensor board 
  // to inform the system of the new locations
  class GPSMessage extends RobotSimEvents::EventMessage
  {
    double latitude;
    double longitude;
    double x;
    double y;
    int precision;
    int satNum;

    int RequestID;
  };

  // LRFReport: Sent from BeoLRF to inform the system of new LRF readings
  class LRFMessage extends RobotSimEvents::EventMessage
  {
    // The distances from each angle in meters
    DoubleSeq distances;	

    // The angle of each laser query. These angles should not change from message
    // to message, but are here just for clarity.
    DoubleSeq angles;

    int RequestID;

    // LRF identification number (for multiple LRFs)
    int LRFID; 
  };

  struct PointCloud 
  {
    ImageIceMod::Point3DIce point;        // x,y,z location
    ImageIceMod::ByteSeq pixel;           // color of point
  };

  sequence<PointCloud> PointCloudSeq;

  // PointCloudReport: Sent from BeoPointCloud to inform the system of new PointCloud readings
  class PointCloudMessage extends RobotSimEvents::EventMessage
  {
    PointCloudSeq pcs;	
    int RequestID;

    // PointCloud identification number (for multiple PC source)
    int PointCloudID; 
  };

  // SLAMReport: Sent from BeoSLAM to inform the system of new SLAM computations
  class SLAMMessage extends RobotSimEvents::EventMessage
  {
    // The map created by SLAM
    //map
    
    // The resolution of the map
    double mapResolution;
    
    // The most likely position of the robot within this map
    double robotX;
    double robotY;
    double robotOri;

    int RequestID;
  };

  // IMU message 
  class IMUMessage extends RobotSimEvents::EventMessage 
  {
    bool validAccAndAng;
    float accelX;
    float accelY;
    float accelZ;
    float angRateX;
    float angRateY;
    float angRateZ;

    bool validRollPitchYaw;
    float roll;
    float pitch;
    float yaw;

    bool validMagnetometer;
    float magX;
    float magY;
    float magZ;

    int RequestID;
  };

  // Message containing camera image and related information
  class CameraMessage extends RobotSimEvents::EventMessage 
  {
    ImageIceMod::ImageIce image;

    int RequestID;

    // camera identification number (fr multiple cameras)
    int cameraID;
  };

  // Message containing camera image related information
  class BlankCameraMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;

    // camera identification number (fr multiple cameras)
    int cameraID;
    // Different type of camer such as RGB or RGBD
    int cameraType;
  };

  // Message containing color + depth camera image 
  // and related information
  class ColorDepthCameraMessage extends RobotSimEvents::EventMessage 
  {
    ImageIceMod::ImageIce colorImage;
    ImageIceMod::ImageIce depthImage;

    int RequestID;

    // camera identification number (fr multiple cameras)
    int cameraID;

  };

  // Message containing screenshot image from BeoApps
  class VisualizerMessage extends RobotSimEvents::EventMessage 
  {
    ImageIceMod::ImageIce image;

    int RequestID;
    int BeoAppID;      
  };


  // Message current accumulated odometry information
  class AccumulatedOdometryMessage extends RobotSimEvents::EventMessage 
  {
    int   RequestID;
    float AccOdo;
  };

  // Message current location reset
  class CurrentLocationResetMessage extends RobotSimEvents::EventMessage 
  {
    int   RequestID;
    int   snum;
    float ltrav;
  };

  /////////////////////////////////////////////////////////////////////////////////////////
  // Gist-Saliency Localization Related Messages
  /////////////////////////////////////////////////////////////////////////////////////////

  struct SalientRegion
  {
    ImageIceMod::Point2DIce salpt;        // salient point
    ImageIceMod::RectangleIce objRect;    // boundary of the salient region
    DoubleSeq salFeatures;                // salient feature vectors
  };

  sequence<SalientRegion> SalientRegionSeq;

  // Message containing vision localization result
  class VisionLocalizationMessage extends RobotSimEvents::EventMessage 
  {
    int   RequestID;
    int   segnum;
    float lentrav;
    float x;
    float y;
  };


  // Message containing gist vector and salient regions,
  // and conspicuity maps
  class GistSalMessage extends RobotSimEvents::EventMessage 
  {
    ImageIceMod::ImageIce currIma;

    DoubleSeq gistFeatures;
    SalientRegionSeq salientRegions;     

    ImageIceMod::ImageIceSeq conspicuityMaps; 
    int RequestID;
  };

  struct LandmarkSearchJob
  {
    int inputSalRegID;

    int dbSegNum;    // Which topological segment?
    int dbLmkNum;    // Which landmark on this segment?
    int dbVOStart;   // range of the Salient region
    int dbVOEnd;
  };

  sequence<LandmarkSearchJob> LandmarkSearchJobSeq;

  // Message to a SIFT worker
  class LandmarkSearchQueueMessage extends RobotSimEvents::EventMessage 
  {
    ImageIceMod::ImageIce currIma;
    SalientRegionSeq salientRegions;

    LandmarkSearchJobSeq jobs;

    int RequestID;
  };

  // Message containing gist vector and salient regions,
  // and conspicuity maps 
  // just like GistSalMessage: 
  // this is done to keep search and tracking synchronized
  class LandmarkTrackMessage extends RobotSimEvents::EventMessage 
  {
    ImageIceMod::ImageIce currIma;

    DoubleSeq gistFeatures;
    SalientRegionSeq salientRegions;

    ImageIceMod::ImageIceSeq conspicuityMaps; 

    bool resetTracker;

    int RequestID;
  };

  // Message from a landmark database search worker
  class LandmarkSearchStatMessage extends RobotSimEvents::EventMessage 
  {
    int inputSalRegID;
    int numObjSearch;
    bool found;	

    int RequestID;
  };

  // Message from a landmark database search worker 
  class LandmarkMatchResultMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;

    ImageIceMod::ImageIce voMatchImage;	
    LandmarkSearchJob matchInfo;

    int segNumMatch;
    float lenTravMatch;

    float MatchScore;
  };

  // Message from a landmark database search master
  class LandmarkDBSearchResultMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;
 
    LandmarkSearchJobSeq matches;
  };

  // Message from a landmark database search worker 
  class CurrentLocationMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;

    int segNum;
    float lenTrav;

    // can add more about the certainty of the results, etc.
  };

  // Tell the SIFT workers to cancel their search
  class CancelSearchMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;
    bool abort;
  };

  // Tell the SIFT workers to cancel their search
  class SearchDoneMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;
  };

  // Tell the GistSalMaster to go to next frame
  class NextFrameMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;
  };

  // Tell the SIFT Master to abort
  class AbortMessage extends RobotSimEvents::EventMessage 
  {
    int RequestID;
  };

  struct MatchCoord
  {
    ImageIceMod::Point2DIceDouble targetCoord; 
    ImageIceMod::Point2DIceDouble dbCoord; 
  };
  sequence<MatchCoord> MatchCoordSeq;


  // Corner procedure computed message
  class CornerMatchMessage extends RobotSimEvents::EventMessage
  {
    double transVel;
    double rotVel;
    int    status;

    MatchCoordSeq matches;
  };

  // GistSal Navigation procedure computed message
  class GSNavMatchMessage extends RobotSimEvents::EventMessage
  {
    double transVel;
    double rotVel;

    MatchCoordSeq matches;
  };


  // Message traversal map
  class TraversalMapMessage extends RobotSimEvents::EventMessage 
  {
    ImageIceMod::ImageIce tmap;
    double heading;
    float  lateralDeviation;
    int RequestID;
  };

  // GoalLocationRequest: ask the BeoNavigator module to set new goal location
  class GoalLocationRequest extends RobotSimEvents::EventMessage
  {
    ImageIceMod::Point2DIce goalLoc;
    
    int  goalType;
    int  snum;
    float ltrav;
  };

  /////////////////////////////////////////////////////////////////////////////////////////
  // Human Interaction Related Messages
  /////////////////////////////////////////////////////////////////////////////////////////

  sequence<ImageIceMod::RectangleIce> RectangleIceSeq;

  // Message all the faces found 
  class FacesMessage extends RobotSimEvents::EventMessage 
  {
    RectangleIceSeq faces;  

     int RequestID;
  };
  
  // All message to speech synthesizer
  class GUISpeechMessage extends RobotSimEvents::EventMessage
  {
    int RequestID;
    string itsMessage;
    int itsCommand;
  };

};

#endif
