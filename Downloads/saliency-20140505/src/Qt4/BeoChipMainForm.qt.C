/*!@file Qt4/BeoChipMainForm.qt.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt4/BeoChipMainForm.qt.C $
// $Id: BeoChipMainForm.qt.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Qt4/BeoChipMainForm.qt.H"

// Qt3 to Qt4 transition notes: the code here was copied from the old Qt/BeoChipMainForm.ui.h
// Also see here for tips: http://qt.nokia.com/doc/4.6/porting4-designer.html

// ######################################################################
BeoChipMainForm::BeoChipMainForm( QWidget* parent, Qt::WindowFlags fl ) :
  QDialog(parent, fl)
{
  setupUi(this);
}

// ######################################################################
BeoChipMainForm::~BeoChipMainForm()
{ }

// ######################################################################
void BeoChipMainForm::init(ModelManager* mgr, nub::soft_ref<BeoChip> thebc)
{
    manager = mgr;
    bc = thebc;

    // make sure we report the correct serial device in use:
    labelSerDev->setText((std::string("Serial Dev: ") +
        bc->getModelParamString("BeoChipDeviceName")).c_str());

    // reset the BeoChip and our displays:
    bc->resetChip();

    // start a timer:
    startTimer(25);
}

// ######################################################################
void BeoChipMainForm::beoChipReset()
{
    LINFO("Resetting form...");
    sliderSSC0->setValue(127); servo[0] = 127;
    sliderSSC1->setValue(127); servo[1] = 127;
    sliderSSC2->setValue(127);        servo[2] = 127;
    sliderSSC3->setValue(127); servo[3] = 127;
    sliderSSC4->setValue(127); servo[4] = 127;
    sliderSSC5->setValue(127); servo[5] = 127;
    sliderSSC6->setValue(127); servo[6] = 127;
    sliderSSC7->setValue(127); servo[7] = 127;
    checkBoxOut0->setChecked(false); dout[0] = false;
    checkBoxOut1->setChecked(false); dout[1] = false;
    checkBoxOut2->setChecked(false); dout[2] = false;
    checkBoxOut3->setChecked(false); dout[3] = false;
    checkDebounce->setChecked(true); debounce = false;
    for (int i = 0; i < 5; i ++) din[i] = true;
    for (int i = 0; i < 2; i ++) { ain[i] = 0; pin[i] = 0; }

    // get BeoChip display going:
    bc->lcdClear();   // 01234567890123456789
    bc->lcdPrintf(0, 0, "test-BeoChipQt   1.0");
    bc->lcdSetAnimation(2);
    bc->captureAnalog(0, true);
    bc->captureAnalog(1, true);
    bc->debounceKeyboard(true);
    bc->captureKeyboard(true);
    bc->capturePulse(0, true);
    bc->capturePulse(1, true);
    for (int i = 0; i < 7; i ++) bc->setServoRaw(i, 127);
    for (int i = 0; i < 3; i ++) bc->setDigitalOut(i, false);
}

// ######################################################################
void BeoChipMainForm::showDin(const int num, const bool state)
{
    if (num < 0 || num > 4) LERROR("Invalid digital input number %d ignored", num);
    else din[num] = state;
}

// ######################################################################
void BeoChipMainForm::showAnalog(const int num, const int val)
{
    if (num < 0 || num > 1) LERROR("Invalid analog input number %d ignored", num);
    else ain[num] = val;
}

// ######################################################################
void BeoChipMainForm::showPWM(const int num, const int val)
{
    if (num < 0 || num > 1) LERROR("Invalid PWM input number %d ignored", num);
    else pin[num] = val;
}

// ######################################################################
void BeoChipMainForm::sliderSSC0_valueChanged( int val )
{ servo[0] = val; }

void BeoChipMainForm::sliderSSC1_valueChanged( int val )
{ servo[1] = val; }

void BeoChipMainForm::sliderSSC2_valueChanged( int val )
{ servo[2] = val; }

void BeoChipMainForm::sliderSSC3_valueChanged( int val )
{ servo[3] = val; }

void BeoChipMainForm::sliderSSC4_valueChanged( int val )
{ servo[4] = val; }

void BeoChipMainForm::sliderSSC5_valueChanged( int val )
{ servo[5] = val; }

void BeoChipMainForm::sliderSSC6_valueChanged( int val )
{ servo[6] = val; }

void BeoChipMainForm::sliderSSC7_valueChanged( int val )
{ servo[7] = val; }

// ######################################################################
void BeoChipMainForm::radioButtonDec_clicked()
{
    radioButtonHex->setChecked(FALSE);
    radioButtonDec->setChecked(TRUE);
    lCDNumberSSC0->setDecMode();
    lCDNumberSSC1->setDecMode();
    lCDNumberSSC2->setDecMode();
    lCDNumberSSC3->setDecMode();
    lCDNumberSSC4->setDecMode();
    lCDNumberSSC5->setDecMode();
    lCDNumberSSC6->setDecMode();
    lCDNumberSSC7->setDecMode();
    lCDNumberA0->setDecMode();
    lCDNumberA1->setDecMode();
    lCDNumberPWM0->setDecMode();
    lCDNumberPWM1->setDecMode();
}

// ######################################################################
void BeoChipMainForm::radioButtonHex_clicked()
{
    radioButtonHex->setChecked(TRUE);
    radioButtonDec->setChecked(FALSE);
    lCDNumberSSC0->setHexMode();
    lCDNumberSSC1->setHexMode();
    lCDNumberSSC2->setHexMode();
    lCDNumberSSC3->setHexMode();
    lCDNumberSSC4->setHexMode();
    lCDNumberSSC5->setHexMode();
    lCDNumberSSC6->setHexMode();
    lCDNumberSSC7->setHexMode();
    lCDNumberA0->setHexMode();
    lCDNumberA1->setHexMode();
    lCDNumberPWM0->setHexMode();
    lCDNumberPWM1->setHexMode();
}

// ######################################################################
void BeoChipMainForm::checkDebounce_stateChanged( int s )
{ if (s) debounce = true; else debounce = false; }

// ######################################################################
void BeoChipMainForm::pushButtonReset_clicked()
{
LINFO("Reset button clicked!");
    bc->resetChip(); // our BeoChipListener should in turn call resetBeoChip on us
}

// ######################################################################
void BeoChipMainForm::checkBoxOut3_stateChanged( int s )
{ if (s) dout[3] = true; else dout[3] = false; }

void BeoChipMainForm::checkBoxOut2_stateChanged( int s )
{ if (s) dout[2] = true; else dout[2] = false; }

void BeoChipMainForm::checkBoxOut1_stateChanged( int s )
{ if (s) dout[1] = true; else dout[1] = false; }

void BeoChipMainForm::checkBoxOut0_stateChanged( int s )
{ if (s) dout[0] = true; else dout[0] = false; }

// ######################################################################
void BeoChipMainForm::timerEvent(QTimerEvent *e)
{
    // first, update the BeoChip:
    for (int i = 0; i < 8; i ++) bc->setServoRaw(i, servo[i]);
    for (int i = 0; i < 4; i ++) bc->setDigitalOut(i, dout[i]);
    bc->debounceKeyboard(debounce);

    // also update our displays:
    digin0->display(din[0] ? 1 : 0);
    digin1->display(din[1] ? 1 : 0);
    digin2->display(din[2] ? 1 : 0);
    digin3->display(din[3] ? 1 : 0);
    digin4->display(din[4] ? 1 : 0);
    lCDNumberA0->display(ain[0]);
    lCDNumberA1->display(ain[1]);
    lCDNumberPWM0->display(pin[0]);
    lCDNumberPWM1->display(pin[1]);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
