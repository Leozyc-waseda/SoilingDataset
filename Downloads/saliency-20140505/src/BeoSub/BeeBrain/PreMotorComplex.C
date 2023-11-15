#ifndef __PREMOTOR_COMPLEX_C__
#define __PREMOTOR_COMPLEX_C__

#include "Util/log.H"
#include "BeoSub/BeeBrain/PreMotorComplex.H"
#include "Raster/Raster.H"


// ######################################################################
PreMotorComplex::PreMotorComplex(nub::soft_ref<SubController> motor, /*rutz::shared_ptr<PreFrontalCortexAgent pfc,*/ std::string name)
//, rutz::shared_ptr<AgentManagerA> managerPtr)
  :
itsCurrentMove(),
itsNextMove(),
itsPrimitiveMotor(motor)
  //itsCortex(pfc)
{
  itsName = name;
  //managerA = NULL;//managerPtr;
  //cortex = NULL; //managerA->getPreFrontalCortexAgent();
  itsStop = itsInterrupt = false;
}

// ######################################################################
PreMotorComplex::PreMotorComplex(std::string name)
//, rutz::shared_ptr<AgentManagerA> managerPtr)
  :
itsCurrentMove(),
itsNextMove()
{
  itsName = name;
  //managerA = NULL;//managerPtr;
  //cortex = NULL; //managerA->getPreFrontalCortexAgent();
  itsStop = itsInterrupt = false;
}

// ######################################################################
PreMotorComplex::~PreMotorComplex() {
}

// ######################################################################
void PreMotorComplex::start() {

  while(!itsStop) {

    while(itsCurrentMove.is_valid() && !itsInterrupt) {
      LINFO("RUNNING MOVE");
      run();
    }

    //    itsPrimitiveMotor->setLevel();

    if(itsInterrupt) {
      LINFO("INTERRUPT MOVE SET");

      if(itsNextMove.is_valid()) {
        LINFO("NEW MOVE SET TO RUN");
        itsCurrentMove = itsNextMove;
        itsNextMove.reset();
        itsInterrupt = false;
      }
      else
        LINFO("NEW MOVE IS NOT VALID. NO MOVE");
    }
    else
      LINFO("NO MOVE");
  }


  LINFO("STOP EVERYTHING");

  itsPrimitiveMotor->setDepth(0);
  while(itsPrimitiveMotor->getDepth() > 0) {}
  itsPrimitiveMotor->killMotors();
}


// ######################################################################
void PreMotorComplex::run(rutz::shared_ptr<ComplexMovement> c) {
  //itsNextMove = c;
  //itsInterrupt = false;
  itsCurrentMove = c;
  //  msgHalt();
  run();
}


// ######################################################################
void PreMotorComplex::run() {

  uint i = 0;

  while(!itsInterrupt && itsCurrentMove.is_valid()) {


    ////////// once function calls ///////
    ////////// world/sensor moves /////////
    i = 0;
    while(!itsInterrupt && i < itsCurrentMove->W_onceFuncs.size()) {
      (*this.*(itsCurrentMove->W_onceFuncs[i]))(itsCurrentMove->W_onceParams[i]);
      i++;
    }

    ////////// vision moves ///////////
    i = 0;
    while(!itsInterrupt && i < itsCurrentMove->Vis_onceFuncs.size()) {
      (*this.*(itsCurrentMove->Vis_onceFuncs[i]))(itsCurrentMove->Vis_onceParams[i]);
      i++;
    }



    /////////// repeat function calls ////////////
    /////////// world/sensor moves ////////////
    i = 0;
    while(!itsInterrupt && i < itsCurrentMove->W_reFuncs.size()) {
      (*this.*(itsCurrentMove->W_reFuncs[i]))(itsCurrentMove->W_reParams[i]);
      i++;
    }

    //////////// vision moves //////////
    i = 0;
    while(!itsInterrupt && i < itsCurrentMove->Vis_reFuncs.size()) {
      (*this.*(itsCurrentMove->Vis_reFuncs[i]))(itsCurrentMove->Vis_reParams[i]);
      i++;
    }

    //////// clear the once function lists //////
    itsCurrentMove->W_onceFuncs.clear();
    itsCurrentMove->W_onceParams.clear();
    itsCurrentMove->Vis_onceFuncs.clear();
    itsCurrentMove->Vis_onceParams.clear();

    if(itsCurrentMove->Vis_reFuncs.size() == 0 && itsCurrentMove->W_reFuncs.size() == 0)
      break;

  }

  // if(!itsInterrupt)
  //  itsCortex->msgMovementComplete();

}




// #####  world/sensor moves #####

// ######################################################################
void PreMotorComplex::turn(const SensorInput goal) {

  if(goal.angle.getVal() != INVALID) {
    LINFO("turn to Aboslute Angle: %f", goal.angle.getVal());
    LINFO("current Heading: %d\n", itsPrimitiveMotor->getHeading());
    itsPrimitiveMotor->setHeading((int)goal.angle.getVal());
  }

}


// ######################################################################
void PreMotorComplex::forward(const SensorInput goal) {

  if(INVALID != goal.data) {
    LINFO("move forward/reverse to dist(m): %f", goal.data);
    //    int speed;
    //    itsPrimitiveMotor->setSpeed(100);
    sleep(1);
    //    itsPrimitiveMotor->setSpeed(100);
    //printf("get current heading: %d\n", itsPrimitiveMotor->getHeading());
  }


}


// ######################################################################
void PreMotorComplex::dive(const SensorInput goal) {

  if(INVALID != goal.data) {
    LINFO("dive/surface to depth(m): %f", goal.data);
    LINFO("curent depth: %d", itsPrimitiveMotor->getDepth());
    itsPrimitiveMotor->setDepth((int)goal.data);
  }


}


// #####  vision moves  #####

// ######################################################################
void PreMotorComplex::vis_turn(const VisionInput& goal) {

  if(goal.position.is_valid())
    printf("turn to pixel coordinate: %d, %d, %d\n",
           goal.position->x, goal.position->y, goal.position->z);
  else if(goal.angle.is_valid())
    printf("turn to camera angle: %f\n", goal.angle->getVal());
}


// ######################################################################
void PreMotorComplex::vis_forward(const VisionInput& goal) {
  if(goal.position.is_valid())
    printf("move forward to pixel coordinate: %d, %d, %d\n",
           goal.position->x, goal.position->y, goal.position->z);
}


// ######################################################################
void PreMotorComplex::vis_dive(const VisionInput& goal) {
  if(goal.position.is_valid())
    printf("dive to pixel coordinate: %d, %d, %d\n",
           goal.position->x, goal.position->y, goal.position->z);

}


// ######################################################################
void PreMotorComplex::vis_center(const VisionInput& goal) {
  if(goal.position.is_valid()) {
    LINFO("center on image coordinate: %d, %d, %d\n",
          goal.position->x, goal.position->y, goal.position->z);
    vis_turn(goal);
    if(goal.position->z != INVALID)
      vis_dive(goal);
    else
      vis_forward(goal);
  }
}


// ######################################################################
void PreMotorComplex::vis_lost(const VisionInput& goal) {
  if(goal.position.is_valid()) {

    LINFO("retract: find lost object, was last seen: %d, %d, %d\n",
          goal.position->x, goal.position->y, goal.position->z);

    //vis_turn(-1*goal);
    //if(goal.position->z != INVALID)
    //  vis_dive(-1*goal);
    //else
    //  vis_forward(-1*goal);
  }
}

#endif
