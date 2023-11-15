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

#include "GUI/DebugWin.H"

void RobotHeadForm::fileNew()
{

}


void RobotHeadForm::fileOpen()
{

}


void RobotHeadForm::fileSave()
{

}


void RobotHeadForm::fileSaveAs()
{

}


void RobotHeadForm::filePrint()
{

}


void RobotHeadForm::fileExit()
{

}


void RobotHeadForm::editUndo()
{

}


void RobotHeadForm::editRedo()
{

}


void RobotHeadForm::editCut()
{

}


void RobotHeadForm::editPaste()
{

}


void RobotHeadForm::editFind()
{

}


void RobotHeadForm::helpIndex()
{

}


void RobotHeadForm::helpContents()
{

}


void RobotHeadForm::helpAbout()
{

}


void RobotHeadForm::init( ModelManager &mgr, nub::soft_ref<BeoHead> beoHead )
{

    itsMgr = &mgr;
    itsBeoHead = beoHead;
  //  itsDispThread = new DispThread;
  //  itsDispThread->init(gb, itsRightEyeDisp, itsLeftEyeDisp);
}


void RobotHeadForm::grab()
{
 // itsDispThread->start();
}

void RobotHeadForm::moveLeftEyePan( int pos )
{
  float moveVal = (float)pos/100.0;
  leftEyePanDisp->setText(QString("%1").arg(moveVal));
  itsBeoHead->setLeftEyePan(moveVal);

}


void RobotHeadForm::moveLeftEyeTilt( int pos )
{
  float moveVal = (float)pos/100.0;
  leftEyeTiltDisp->setText(QString("%1").arg(moveVal));
  itsBeoHead->setLeftEyeTilt(moveVal);

}

void RobotHeadForm::moveRightEyePan( int pos )
{
  float moveVal = (float)pos/100.0;
  leftEyePanDisp->setText(QString("%1").arg(moveVal));
  itsBeoHead->setRightEyePan(moveVal);

}


void RobotHeadForm::moveRightEyeTilt( int pos )
{
  float moveVal = (float)pos/100.0;
  leftEyeTiltDisp->setText(QString("%1").arg(moveVal));
  itsBeoHead->setRightEyeTilt(moveVal);

}

void RobotHeadForm::moveHeadPan( int pos )
{
  float moveVal = (float)pos/100.0;
  headPanDisp->setText(QString("%1").arg(moveVal));
  itsBeoHead->setHeadPan(moveVal);

}

void RobotHeadForm::moveHeadTilt( int pos )
{
  float moveVal = (float)pos/100.0;
  headTiltDisp->setText(QString("%1").arg(moveVal));
  itsBeoHead->setHeadTilt(moveVal);

}


void RobotHeadForm::moveHeadYaw( int pos )
{
  float moveVal = (float)pos/100.0;
  headYawDisp->setText(QString("%1").arg(moveVal));
  itsBeoHead->setHeadYaw(moveVal);

}



void RobotHeadForm::relaxNeck()
{

  itsBeoHead->relaxNeck();
}


void RobotHeadForm::restPos()
{
  itsBeoHead->moveRestPos();

}
