/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you want to add, delete, or rename functions or slots, use
** Qt Designer to update this file, preserving your code.
**
** You should not define a constructor or destructor in this file.
** Instead, write your code in functions called init() and destroy().
** These will automatically be called by the form's constructor and
** destructor.
*****************************************************************************/


void RT100ControlForm::init( ModelManager * mgr, nub::soft_ref<RT100> rt100 )
{

  itsMgr = mgr;
  itsRT100 = rt100;
}


void RT100ControlForm::initArm()
{

  armStatusMsg->setText(QString("Initalizing arm..."));
  itsRT100->init();
  armStatusMsg->setText(QString("Arm initalized"));


}


void RT100ControlForm::homeArm()
{

  itsRT100->gotoHomePosition();
}


void RT100ControlForm::moveArm()
{
  LINFO("Moving arm\n");
  itsRT100->moveArm(true); //wait till move is done
  LINFO("Done moving arm\n");
}


void RT100ControlForm::moveZed( int val )
{
  itsRT100->setJointPosition(RT100::ZED, val, moveMode->isChecked());
}


void RT100ControlForm::moveSholder( int val )
{

  itsRT100->setJointPosition(RT100::SHOLDER, val, moveMode->isChecked());
}


void RT100ControlForm::moveElbow( int val )
{

  itsRT100->setJointPosition(RT100::ELBOW, val, moveMode->isChecked());
}


void RT100ControlForm::moveYaw( int val )
{

  itsRT100->setJointPosition(RT100::YAW, val, moveMode->isChecked());
}


void RT100ControlForm::moveWrist1( int val )
{

  itsRT100->setJointPosition(RT100::WRIST1, val, moveMode->isChecked());
}


void RT100ControlForm::moveWrist2( int val )
{

  itsRT100->setJointPosition(RT100::WRIST2, val, moveMode->isChecked());
}


void RT100ControlForm::moveGripper( int val )
{

  itsRT100->setJointPosition(RT100::GRIPPER, val, moveMode->isChecked());
}

void RT100ControlForm::getCurrentJointPositions()
{
  short int val;

  itsRT100->getJointPosition(RT100::ZED, &val);
  zedVal->setValue(val);
  LDEBUG("Zed at %i", val);

  itsRT100->getJointPosition(RT100::ELBOW, &val);
  elbowVal->setValue(val);
  LDEBUG("Elbow at %i", val);

  itsRT100->getJointPosition(RT100::SHOLDER, &val);
  sholderVal->setValue(val);
  LDEBUG("Sholder at %i", val);


  itsRT100->getJointPosition(RT100::YAW, &val);
  yawVal->setValue(val);
  LDEBUG("Yaw at %i", val);

  itsRT100->getJointPosition(RT100::GRIPPER, &val);
  gripperVal->setValue(val);
  LDEBUG("Gripper at %i", val);


  int wrist1Pos, wrist2Pos;
  itsRT100->getJointPosition(RT100::WRIST1, &val);
  wrist1Val->setValue(val);
  wrist1Pos = val;
  LDEBUG("Wrist1 at %i", val);

  itsRT100->getJointPosition(RT100::WRIST2, &val);
  wrist2Val->setValue(val);
  wrist2Pos = val;
  LDEBUG("Wrist2 at %i", val);


}


void RT100ControlForm::wristRoll( int val )
{
  itsRT100->setJointPosition(RT100::ROLL_WRIST, val, moveMode->isChecked());
}


void RT100ControlForm::wristTilt( int val )
{
  itsRT100->setJointPosition(RT100::TILT_WRIST, val, moveMode->isChecked());
}



void RT100ControlForm::doInterpolation()
{


  std::vector<short int> moveVals(itsRT100->getNumJoints(), 0);

  moveVals[RT100::ZED] = zedInterpolation->value();
  moveVals[RT100::SHOLDER] = sholderInterpolation->value();
  moveVals[RT100::ELBOW] = elbowInterpolation->value();
  moveVals[RT100::YAW] = yawInterpolation->value();
  moveVals[RT100::WRIST1] = wrist1Interpolation->value();
  moveVals[RT100::WRIST2] = wrist2Interpolation->value();
  moveVals[RT100::GRIPPER] = gripperInterpolation->value();

  itsRT100->interpolationMove(moveVals);

}
