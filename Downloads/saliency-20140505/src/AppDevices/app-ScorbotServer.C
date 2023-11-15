#include <Ice/Ice.h>
#include "Ice/Scorbot.ice.H"
#include "Component/ModelComponent.H"
#include "Component/ModelManager.H"
#include "Devices/Scorbot.H"
#include <signal.h>

// ######################################################################
const ModelOptionCateg MOC_ScorbotServer = 
{ MOC_SORTPRI_3, "Scorbot Server Related Options" };

const ModelOptionDef OPT_Port = 
{ MODOPT_ARG(std::string), "Port", &MOC_ScorbotServer, OPTEXP_CORE,
  "Scorbot Server Port Number",
  "scorbot-port", '\0', "", "10000" };

class ScorbotI: public Robots::ScorbotIce, public ModelComponent
{
  public:
    ScorbotI(OptionManager& mgr,
        const std::string& descrName = "ScorbotServer",
        const std::string& tagName = "ScorbotServer") :
      ModelComponent(mgr, descrName, tagName),
      itsScorbot(new Scorbot(mgr)), 
      itsPort(&OPT_Port, this, 0)
  { 
    addSubComponent(itsScorbot);
  }

    void init()
    {
	    //Initial arm position will be 0-0
            Scorbot::ArmPos armPos;
	    armPos.base = 0;
	    armPos.sholder = 0;
	    armPos.elbow = 0;
	    armPos.wristRoll = 0;
	    armPos.wristPitch = 0;
	    armPos.ex1 =0;
	    armPos.ex2 =0;
	    itsScorbot->resetEncoders();
	    itsScorbot->setArmPos(armPos);
	    itsScorbot->motorsOn();

    }

    // ######################################################################
    //!Get the inverse kinematics
    Robots::ArmPos getIK(float x, float y, float z, const Ice::Current&)
    {
      Point3D<float> pos(x, y, z);

      Scorbot::ArmPos ikarmPos = itsScorbot->getIKArmPos(pos);

      Robots::ArmPos ikPosIce;
      ikPosIce.base = ikarmPos.base;
      ikPosIce.shoulder = ikarmPos.sholder;
      ikPosIce.elbow = ikarmPos.elbow;
      ikPosIce.gripper = ikarmPos.gripper;
      ikPosIce.wristroll = ikarmPos.wristRoll;
      ikPosIce.wristpitch = ikarmPos.wristPitch;
      ikPosIce.wrist1 = ikarmPos.wrist1;
      ikPosIce.wrist2 = ikarmPos.wrist2;
      ikPosIce.ex1 = ikarmPos.ex1;
      ikPosIce.ex2 = ikarmPos.ex2;
      return ikPosIce;
    }

    // ######################################################################
    //!Set the end effector position in x,y,z
    bool getEFpos(float &x, float &y, float &z, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      itsScorbot->getEF_pos(x, y, z);
      return true;
    }

    // ######################################################################
    //!get the end effector position in x,y,z
    bool setEFPos(float x,float y,float z, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      Scorbot::ArmPos IKpos = itsScorbot->getIKArmPos(Point3D<float>(x,y,z));
      itsScorbot->setArmPos(IKpos);
      return true;
    }

    // ######################################################################
    //! Set the motor pwm 
    bool setMotor(Robots::JOINTS joint, int pwm, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      return itsScorbot->setMotor(RobotArm::MOTOR(joint), pwm);
    }

    // ######################################################################
    //! Get the current pwm value
    int getPWM(Robots::JOINTS j, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      return itsScorbot->getPWM(RobotArm::MOTOR(j));
    }

    // ######################################################################
    //! Set the joint to a given position
    bool setJointPos(Robots::JOINTS joint, int pos, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      return itsScorbot->setJointPos(RobotArm::MOTOR(joint), pos);
    }

    // ######################################################################
    //! Get the joint position
    int getJointPos(Robots::JOINTS joint, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      return itsScorbot->getJointPos(RobotArm::MOTOR(joint));
    }

    // ######################################################################
    //! Get the anguler joint position
    float getEncoderAng(Robots::JOINTS joint, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      return itsScorbot->getEncoderAng(RobotArm::MOTOR(joint));
    }

    // ######################################################################
    //! Reset all encoders to 0
    void resetEncoders(const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      itsScorbot->resetEncoders();
    }

    // ######################################################################
    //! Stop eveything
    void stopAllMotors(const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      itsScorbot->stopAllMotors();
    }

    // ######################################################################
    void setSafety(bool val, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      itsScorbot->setSafety(val);
    }

    // ######################################################################
    //! Home Joint
    void homeMotor(Robots::JOINTS joint, int LimitSeekSpeed, int MSJumpSpeed,
        float MSJumpDelay, int MSSeekSpeed, bool MSStopCondition,
        bool checkMS, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      itsScorbot->homeMotor(RobotArm::MOTOR(joint), LimitSeekSpeed, MSJumpSpeed, 
          MSJumpSpeed, MSSeekSpeed, MSSeekSpeed, checkMS);
    }

    // ######################################################################
    //! Home All Joints
    void homeMotors(const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      itsScorbot->homeMotors();
    }

    // ######################################################################
    //! Get the microSwitch states
    int getMicroSwitch(const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      return itsScorbot->getMicroSwitch();
    }

    // ######################################################################
    //! Get the microswitch to a spacific joint
    int getMicroSwitchMotor(Robots::JOINTS m, const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      return itsScorbot->getMicroSwitchMotor(RobotArm::MOTOR(m));
    }

    // ######################################################################
    //! Shutdown
    void shutdown(const Ice::Current&)
    {
      std::cout << __FUNCTION__ << std::endl;
      prepareForExit();
      exit(0);
    }

    void motorsOn(const Ice::Current&)
    {
      itsScorbot->motorsOn();
    }

    void motorsOff(const Ice::Current&)
    {
      itsScorbot->motorsOff();
    }


    // ######################################################################
    //! Convert encoder tics to angle
    double enc2ang(int encoderTicks, const Ice::Current&)
    {
      return itsScorbot->enc2ang(encoderTicks);
    }

    // ######################################################################
    //! Convert angle to encoder ticks
    int ang2enc(double degrees, const Ice::Current&)
    {
      return itsScorbot->ang2enc(degrees);
    }

    // ######################################################################
    //! Convert encoder ticks to mm
    double enc2mm(int encoderTicks, const Ice::Current&)
    {
      return itsScorbot->enc2mm(encoderTicks);
    }

    // ######################################################################
    //! Convert mm to encoder tics
    int mm2enc(double mm, const Ice::Current&)
    {
      return itsScorbot->mm2enc(mm);
    }

    bool setArmPos(const Robots::ArmPos& pos, const Ice::Current&)
    {
      Scorbot::ArmPos armPos;
      armPos.base = pos.base;
      armPos.sholder = pos.shoulder;
      armPos.elbow = pos.elbow;
      armPos.gripper = pos.gripper;
      armPos.wristRoll = pos.wristroll;
      armPos.wristPitch = pos.wristpitch;
      armPos.wrist1 = pos.wrist1;
      armPos.wrist2 = pos.wrist2;
      armPos.ex1 = pos.ex1;
      armPos.ex2 = pos.ex2;
      armPos.duration = pos.duration;
      itsScorbot->setArmPos(armPos);
      return true;
    }

    Robots::ArmPos getArmPos(const Ice::Current&)
    {
      Scorbot::ArmPos armPos = itsScorbot->getArmPos();
      Robots::ArmPos posIce;
      posIce.base = armPos.base;
      posIce.shoulder = armPos.sholder;
      posIce.elbow = armPos.elbow;
      posIce.gripper = armPos.gripper;
      posIce.wristroll = armPos.wristRoll;
      posIce.wristpitch = armPos.wristPitch;
      posIce.wrist1 = armPos.wrist1;
      posIce.wrist2 = armPos.wrist2;
      posIce.ex1 = armPos.ex1;
      posIce.ex2 = armPos.ex2;
      return posIce;
    }
    

    std::string getPort() { return itsPort.getVal(); }

    void prepareForExit()
    {
      itsScorbot->stopAllMotors();
      itsScorbot->motorsOff();
      sleep(1);
    }

  private:
    nub::soft_ref<Scorbot> itsScorbot;
    OModelParam<std::string> itsPort;
};
nub::soft_ref<ScorbotI> scorbotServer;

// ######################################################################
void terminate(int s)
{
  std::cerr <<
    std::endl << "*** INTERRUPT - SHUTTING DOWN SCORBOT ***" << std::endl;

  scorbotServer->prepareForExit();
  exit(0);
}

// ######################################################################
int main(int argc, char** argv)
{
	signal(SIGHUP, terminate);  signal(SIGINT, terminate);
	signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
	signal(SIGALRM, terminate);

  ModelManager manager("ScorbotServerManager");
  scorbotServer.reset(new ScorbotI(manager));
  manager.addSubComponent(scorbotServer);
  manager.parseCommandLine(argc, argv, "", 0, 0);
  manager.start();

  std::string connString("default -p ");
  connString += scorbotServer->getPort();

  int status = 0;
  Ice::CommunicatorPtr ic;

  try {
    ic = Ice::initialize(argc, argv);
    Ice::ObjectAdapterPtr adapter = 
      ic->createObjectAdapterWithEndpoints(
          "ScorbotServerAdapter", connString.c_str());
    Ice::ObjectPtr object = scorbotServer.get();
    adapter->add(object, ic->stringToIdentity("ScorbotServer"));
    adapter->activate();
    scorbotServer->init();
    ic->waitForShutdown();
  } catch(const Ice::Exception& e) {
    std::cerr << e << std::endl;
    status = 1;
  } catch(const char* msg)  {
    std::cerr << msg << std::endl;
    status = 1;
  }
  if(ic) {
    try {
      ic->destroy(); 
    } catch(const Ice::Exception& e) {
      std::cerr << e << std::endl;
      status = 1;
    }
  }

  scorbotServer->prepareForExit();
  return status;
}
