/*!@file Qt4/BeoChipMainForm.qt.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt4/BeoChipMainForm.qt.H $
// $Id: BeoChipMainForm.qt.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef QT4_BEOCHIPMAINFORM_QT_H_DEFINED
#define QT4_BEOCHIPMAINFORM_QT_H_DEFINED

#include "Qt4/BeoChipMainForm.ui.H"
#include "Devices/BeoChip.H"
#include "Component/ModelManager.H"

// Qt3 to Qt4 transition notes: the code here was derived from the old src/Qt/ui/BeoChipMainform.h


//! Implementation of slots and signals for the BeoChipMaiForm Qt4 dialog
class BeoChipMainForm : public QDialog, public Ui::BeoChipMainForm
{
  Q_OBJECT

public:
  // Constructor
  BeoChipMainForm( QWidget* parent = 0, Qt::WindowFlags fl = 0 );

  // Destructor
  virtual ~BeoChipMainForm();

public slots:
    virtual void init( ModelManager * mgr, nub::soft_ref<BeoChip> thebc );
    virtual void beoChipReset();
    virtual void showDin( const int num, const bool state );
    virtual void showAnalog( const int num, const int val );
    virtual void showPWM( const int num, const int val );
    virtual void sliderSSC0_valueChanged( int val );
    virtual void sliderSSC1_valueChanged( int val );
    virtual void sliderSSC2_valueChanged( int val );
    virtual void sliderSSC3_valueChanged( int val );
    virtual void sliderSSC4_valueChanged( int val );
    virtual void sliderSSC5_valueChanged( int val );
    virtual void sliderSSC6_valueChanged( int val );
    virtual void sliderSSC7_valueChanged( int val );
    virtual void radioButtonDec_clicked();
    virtual void radioButtonHex_clicked();
    virtual void checkDebounce_stateChanged( int s );
    virtual void pushButtonReset_clicked();
    virtual void checkBoxOut3_stateChanged( int s );
    virtual void checkBoxOut2_stateChanged( int s );
    virtual void checkBoxOut1_stateChanged( int s );
    virtual void checkBoxOut0_stateChanged( int s );
    virtual void timerEvent( QTimerEvent * e );

protected:
    ModelManager *manager;
    nub::soft_ref<BeoChip> bc;

private:
    int servo[8];
    bool dout[4];
    bool debounce;
    bool din[5];
    int ain[2];
    int pin[2];
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // QT4_BEOCHIPMAINFORM_QT_H_DEFINED
