/****************************************************************************
** Form interface generated from reading ui file 'Qt/BeoBotQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef BEOBOTQTMAINFORM_H
#define BEOBOTQTMAINFORM_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qwidget.h>
#include "Util/StringConversions.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class BeoBotMap;
class QLabel;

class BeoBotQtMainForm : public QWidget
{
    Q_OBJECT

public:
    BeoBotQtMainForm( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~BeoBotQtMainForm();

    QLabel* MapLabel;
    QLabel* CameraLabel;
    QLabel* DisplayLabel;
    QLabel* ParameterLabel;
    BeoBotMap* map;
    QLabel* displayCoord;
    QLabel* displayImage3;
    QLabel* displayImage4;
    QLabel* displayImage2;
    QLabel* displayImage1;

    void init();

public slots:
    virtual void displayFunc();
    virtual void timerEvent( QTimerEvent * e );

protected:

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // BEOBOTQTMAINFORM_H
