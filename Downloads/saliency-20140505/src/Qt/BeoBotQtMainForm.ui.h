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
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
void BeoBotQtMainForm::init()
{
    startTimer(100);
}

void BeoBotQtMainForm::displayFunc()
{
        float fx, fy;
        map->returnSelectedCoord(fx, fy);
        displayCoord->setText(QString(sformat("x:%f y:%f", fx, fy)));
}

void BeoBotQtMainForm::timerEvent( QTimerEvent *e )
{
        displayFunc();
}
