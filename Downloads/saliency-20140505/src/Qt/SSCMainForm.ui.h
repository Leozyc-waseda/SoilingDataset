/*! @file Qt/SSCMainForm.ui.h functions relating to SSC control main form */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/SSCMainForm.ui.h $
// $Id: SSCMainForm.ui.h 5769 2005-10-20 16:50:33Z rjpeters $

/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename functions or slots use
** Qt Designer which will update this file, preserving your code. Create an
** init() function in place of a constructor, and a destroy() function in
** place of a destructor.
*****************************************************************************/

void SSCMainForm::init(ModelManager* mgr, nub::soft_ref<SSC> thessc)
{
    manager = mgr;
    ssc = thessc;
    devname = "/dev/ttyS0";
    baud = "9600";
}

void SSCMainForm::lineEditSerDev_textChanged(const QString &newdevname)
{
    // keep the value and wait until <return> is pressed
    devname = newdevname;
}


void SSCMainForm::lineEditBaudrate_textChanged( const QString &newbaud )
{
    // keep the value and wait until <return> is pressed
    baud = newbaud;
}


void SSCMainForm::sliderSSC1_valueChanged( int val )
{
    ssc->moveRaw(0, val);
}


void SSCMainForm::sliderSSC2_valueChanged( int val )
{
    ssc->moveRaw(1, val);
}


void SSCMainForm::sliderSSC3_valueChanged( int val )
{
    ssc->moveRaw(2, val);
}


void SSCMainForm::sliderSSC4_valueChanged( int val )
{
    ssc->moveRaw(3, val);
}


void SSCMainForm::sliderSSC5_valueChanged( int val )
{
    ssc->moveRaw(4, val);
}


void SSCMainForm::sliderSSC6_valueChanged( int val )
{
    ssc->moveRaw(5, val);
}


void SSCMainForm::sliderSSC7_valueChanged( int val )
{
    ssc->moveRaw(6, val);
}


void SSCMainForm::sliderSSC8_valueChanged( int val )
{
    ssc->moveRaw(7, val);
}


void SSCMainForm::radioButtonDec_clicked()
{
    radioButtonHex->setChecked(FALSE);
    radioButtonDec->setChecked(TRUE);
    lCDNumberSSC1->setDecMode();
    lCDNumberSSC2->setDecMode();
    lCDNumberSSC3->setDecMode();
    lCDNumberSSC4->setDecMode();
    lCDNumberSSC5->setDecMode();
    lCDNumberSSC6->setDecMode();
    lCDNumberSSC7->setDecMode();
    lCDNumberSSC8->setDecMode();
}

void SSCMainForm::radioButtonHex_clicked()
{
    radioButtonHex->setChecked(TRUE);
    radioButtonDec->setChecked(FALSE);
    lCDNumberSSC1->setHexMode();
    lCDNumberSSC2->setHexMode();
    lCDNumberSSC3->setHexMode();
    lCDNumberSSC4->setHexMode();
    lCDNumberSSC5->setHexMode();
    lCDNumberSSC6->setHexMode();
    lCDNumberSSC7->setHexMode();
    lCDNumberSSC8->setHexMode();
}


void SSCMainForm::lineEditSerDev_returnPressed()
{
    manager->stop();
    manager->setModelParamString("SerialPortDevName",
                                 devname.ascii(), MC_RECURSE);
    manager->start();
    LINFO("Switched to device %s", devname.ascii());
}


void SSCMainForm::lineEditBaudrate_returnPressed()
{
    manager->stop();
    manager->setModelParamString("SerialPortBaud",
                                 baud.ascii(), MC_RECURSE);
    manager->start();
    LINFO("Switched to baudrate %s", baud.ascii());
}
