#ifndef __PREMOTOR_COMPLEX_C__
#define __PREMOTOR_COMPLEX_C__

#include "Util/log.H"
#include "PreMotorComplex.H"
#include "Raster/Raster.H"


// ######################################################################
PreMotorComplex::PreMotorComplex(nub::soft_ref<SubController> motor, /*rutz::shared_ptr<CaptainAgent pfc,*/ std::string name)
//, rutz::shared_ptr<AgentManagerA> managerPtr)
//   :
// itsCurrentMove(),
// itsNextMove(),
// itsPrimitiveMotor(motor)
  : SubmarineAgent(name)
{
//   itsName = name;
//   //managerA = NULL;//managerPtr;
//   //cortex = NULL; //managerA->getCaptainAgent();
//   itsStop = itsInterrupt = isMoveDone = false;
//   //initial depth value at surface
//   itsPrimitiveMotor->killMotors();
//   itsPrimitiveMotor->setMotorsOn(false);

}

// ######################################################################
PreMotorComplex::PreMotorComplex(std::string name)
  : SubmarineAgent(name)
//, rutz::shared_ptr<AgentManagerA> managerPtr)
//   :
// itsCurrentMove(),
// itsNextMove()
{
  itsName = name;
  //managerA = NULL;//managerPtr;
  //cortex = NULL; //managerA->getCaptainAgent();
//   itsStop = itsInterrupt = false;
}

// ######################################################################
PreMotorComplex::~PreMotorComplex() {
}

// ######################################################################
// void PreMotorComplex::start() {

//   itsPrimitiveMotor->setMotorsOn(false);
//   while(!itsPrimitiveMotor->getKillSwitch()) {
//     sleep(1);
//     LINFO("Waiting for Kill Switch");
//   }


//   while(itsPrimitiveMotor->getKillSwitch() && !itsStop) {

//     while(itsCurrentMove.is_valid() && !itsInterrupt) {
//       LINFO("RUNNING MOVE: %d & %d",
//             itsCurrentMove.is_valid(), !itsInterrupt);
//       run();
//     }

//     //    itsPrimitiveMotor->setLevel();

//     if(itsInterrupt) {
//       LINFO("INTERRUPT MOVE SET");

//       if(itsNextMove.is_valid()) {
//         LINFO("NEW MOVE SET TO RUN");
//         itsCurrentMove = itsNextMove;
//         itsNextMove.reset();
//         itsInterrupt = false;
//       }
//       else
//         LINFO("NEW MOVE IS NOT VALID. NO MOVE");
//     }
//     else
//       LINFO("NO MOVE");

//     isMoveDone = false;
//   }


//   LINFO("STOP EVERYTHING");

//   itsCurrentMove.reset();
//   itsNextMove.reset();

//   itsPrimitiveMotor->killMotors();
//   itsPrimitiveMotor->setMotorsOn(false);
// }


// // ######################################################################
// void PreMotorComplex::run(rutz::shared_ptr<ComplexMovement> c) {
//   msgHalt();
//   itsNextMove = c;
//   //itsInterrupt = false;
//   itsCurrentMove.reset();
//   //msgHalt();
//   //run();
// }


// // ######################################################################
// void PreMotorComplex::run() {

// //   uint i = 0;

// //   while(!itsInterrupt && itsCurrentMove.is_valid()) {


// //     ////////// once function calls ///////
// //     ////////// world/sensor moves /////////
// //     i = 0;
// //     while(!itsInterrupt && i < itsCurrentMove->W_onceFuncs.size()) {
// //       (*this.*(itsCurrentMove->W_onceFuncs[i]))(itsCurrentMove->W_onceParams[i]);
// //       i++;
// //     }

// //     ////////// vision moves ///////////
// //     i = 0;
// //     while(!itsInterrupt && i < itsCurrentMove->Vis_onceFuncs.size()) {
// //       (*this.*(itsCurrentMove->Vis_onceFuncs[i]))(itsCurrentMove->Vis_onceParams[i]);
// //       i++;
// //     }


// //     /////////// repeat function calls ////////////
// //     /////////// world/sensor moves ////////////
// //     i = 0;
// //     while(!itsInterrupt && i < itsCurrentMove->W_reFuncs.size()) {
// //       (*this.*(itsCurrentMove->W_reFuncs[i]))(itsCurrentMove->W_reParams[i]);
// //       i++;
// //     }

// //     //////////// vision moves //////////
// //     i = 0;
// //     while(!itsInterrupt && i < itsCurrentMove->Vis_reFuncs.size()) {
// //       (*this.*(itsCurrentMove->Vis_reFuncs[i]))(itsCurrentMove->Vis_reParams[i]);
// //       i++;
// //     }

// //     if(itsCurrentMove->W_onceFuncs.size() > 0 || itsCurrentMove->Vis_onceFuncs.size() > 0) {
// //       isMoveDone = true;
// //       //////// clear the once function lists //////
// //       itsCurrentMove->W_onceFuncs.clear();
// //       itsCurrentMove->W_onceParams.clear();
// //       itsCurrentMove->Vis_onceFuncs.clear();
// //       itsCurrentMove->Vis_onceParams.clear();
// //     }

// //     if(itsCurrentMove->Vis_reFuncs.size() == 0 &&
// //        itsCurrentMove->W_reFuncs.size() == 0 &&
// //        itsCurrentMove->Vis_onceFuncs.size() == 0 &&
// //        itsCurrentMove->W_onceFuncs.size() == 0)
// //       break;

// //   }

// //   itsInterrupt = true;

// }




// // #####  world/sensor moves #####

// // ######################################################################
// void PreMotorComplex::turn(const SensorInput goal) {

//   if(goal.angle.getVal() != INVALID) {
//     LINFO("turn to Aboslute Angle: %f", goal.angle.getVal());
//     LINFO("current Heading: %d\n", itsPrimitiveMotor->getHeading());
//     itsPrimitiveMotor->setHeading((int)goal.angle.getVal());
//     while(itsPrimitiveMotor->getHeadingErr() > 5)
//       usleep(100000);
//   }

// }


// // ######################################################################
// void PreMotorComplex::forward(const SensorInput goal) {

//   if(INVALID != goal.data) {
//     LINFO("move forward/reverse to dist(m): %f", goal.data);
//     //    int speed;
//     int direction = goal.data > 0 ? 1 : -1;
//     itsPrimitiveMotor->setSpeed(direction * 100);
//     int time = abs((int)goal.data) * VELOCITY_CONST;
//     sleep(time);
//     itsPrimitiveMotor->setSpeed(-1* direction * 100);
//     sleep(1);
//     itsPrimitiveMotor->setSpeed(0);
//     //printf("get current heading: %d\n", itsPrimitiveMotor->getHeading());
//   }


// }


// // ######################################################################
// void PreMotorComplex::dive(const SensorInput goal) {

//   if(INVALID != goal.data) {
//     int deptherr = (int)goal.data;// * DEPTH_ERROR_CONST;


//     LINFO("dive/surface to depth: %i", surfaceDepth);
//     int newDepth = surfaceDepth + deptherr;
//     LINFO("dive/surface to depth(Liors): %i", newDepth);
//     itsPrimitiveMotor->setDepth(newDepth);
//     //    LINFO("Err %i", itsPrimitiveMotor->getDepthErr());
//     sleep(1);
// //     while(itsPrimitiveMotor->getDepthErr() > 3)
// //       {
// //         usleep(100000);
// //         LINFO("curent depth: %d", itsPrimitiveMotor->getDepth());
// //       }
//   }


// }


// void PreMotorComplex::wait(const SensorInput goal) {

//   if(INVALID != goal.data) {
//     LINFO("waiting for %f secs", goal.data);
//     sleep((int)goal.data);
//   }
// }

// void PreMotorComplex::startWait(const SensorInput goal) {

//   if(INVALID != goal.data) {
//     LINFO("waiting for %f secs", goal.data);
//     sleep((int)goal.data);
//   }
//   itsPrimitiveMotor->setMotorsOn(true);
//   surfaceDepth = itsPrimitiveMotor->getDepth();
// }


// // #####  vision moves  #####

// // ######################################################################
// void PreMotorComplex::vis_turn(const VisionInput& goal) {

//   if(goal.position.is_valid() && goal.position->x != INVALID) {
//     int xerr = goal.position->x - 180; // HARDCODED CENTER!!!!!!
//     int desiredHeading = xerr * XAXIS_ERROR_CONST;

//     itsPrimitiveMotor->setHeading(itsPrimitiveMotor->getHeading()
//                                   + desiredHeading);

//     printf("turn to pixel coordinate: %d, %d, %d\n",
//            goal.position->x, goal.position->y, goal.position->z);

//     while(itsPrimitiveMotor->getHeadingErr() > 5)
//       usleep(100000);
//   }
//   else if(goal.angle.is_valid()) {
//     int angerr = (90 - (int)goal.angle->getVal());
//     int desiredHeading = angerr * ANGLE_ERROR_CONST;
//     itsPrimitiveMotor->setHeading(itsPrimitiveMotor->getHeading()
//                                   + desiredHeading);
//     printf("turn to camera angle: %f\n", goal.angle->getVal());

//     while(itsPrimitiveMotor->getHeadingErr() > 5)
//       usleep(100000);
//   }


// }

// // ######################################################################
// void PreMotorComplex::vis_forward(const VisionInput& goal) {
//   if(goal.position.is_valid() && goal.position->y != INVALID) {
//     int yerr = 120 - goal.position->y; /// HARDCODED CENTER!!!!!
//     int desiredSpeed = yerr * YAXIS_ERROR_CONST;

//     itsPrimitiveMotor->setSpeed(desiredSpeed);
//     int time = 100000 * YAXIS_ERROR_CONST;
//     usleep(time);
//     itsPrimitiveMotor->setSpeed(100);
//     //itsPrimitiveMotor->setSpeed(-1 * desiredSpeed);
//     usleep(50000);
//     itsPrimitiveMotor->setSpeed(0);


//     printf("move forward to pixel coordinate: %d, %d, %d\n",
//            goal.position->x, goal.position->y, goal.position->z);
//   }
// }


// // ######################################################################
// void PreMotorComplex::vis_dive(const VisionInput& goal) {
//   if(goal.position.is_valid() && goal.position->z != INVALID) {
//     int zerr = goal.position->z - 120; ///// HARDCODED CENTER!!!!
//     int desiredDepth = zerr * ZAXIS_ERROR_CONST;

//     itsPrimitiveMotor->setDepth(itsPrimitiveMotor->getDepth()
//                                 + desiredDepth);


//     printf("dive to pixel coordinate: %d, %d, %d\n",
//            goal.position->x, goal.position->y, goal.position->z);
// //     while(itsPrimitiveMotor->getDepthErr() > 5)
// //       usleep(100000);
//   }
// }

// // ######################################################################
// void PreMotorComplex::vis_center(const VisionInput& goal) {
//   if(goal.position.is_valid()) {
//     LINFO("center on image coordinate: %d, %d, %d\n",
//           goal.position->x, goal.position->y, goal.position->z);

//     vis_turn(goal);
//     if(goal.position->z != INVALID)
//       vis_dive(goal);
//     else
//       vis_forward(goal);
//   }
// }


// // ######################################################################
// void PreMotorComplex::vis_lost(const VisionInput& goal) {
//   if(goal.position.is_valid()) {

//     LINFO("retract: find lost object, was last seen: %d, %d, %d\n",
//           goal.position->x, goal.position->y, goal.position->z);

//     //vis_turn(-1*goal);
//     //if(goal.position->z != INVALID)
//     //  vis_dive(-1*goal);
//     //else
//     //  vis_forward(-1*goal);
//   }
// }

#endif
