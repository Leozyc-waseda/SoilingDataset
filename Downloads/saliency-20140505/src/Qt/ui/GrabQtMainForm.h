/****************************************************************************
** Form interface generated from reading ui file 'Qt/GrabQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef GRABQTMAINFORM_H
#define GRABQTMAINFORM_H

#include <qvariant.h>
#include <qdialog.h>
#include "Transport/FrameIstream.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "ModelManagerControl.h"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QLabel;
class QSlider;
class QPushButton;

class GrabQtMainForm : public QDialog
{
    Q_OBJECT

public:
    GrabQtMainForm( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~GrabQtMainForm();

    QLabel* tiltLabel;
    QLabel* panLabel;
    QSlider* panSlider;
    QPushButton* pauseButton;
    QPushButton* saveButton;
    QSlider* tiltSlider;

public slots:
    virtual void init( int argc, const char * * argv );
    virtual void handlePauseButton();
    virtual void handlePanSlider( int pos );
    virtual void handleTiltSlider( int pos );
    virtual void saveImage();
    virtual void timerEvent( QTimerEvent * );
    virtual void grabImage();

protected:
    nub::soft_ref<FrameIstream> gb;
    nub::soft_ref<FrameGrabberConfigurator> gbc;
    ModelManager manager;
    QDialog *display;
    bool grabbing;
    ModelManagerControl mmc;


protected slots:
    virtual void languageChange();

};

#endif // GRABQTMAINFORM_H
