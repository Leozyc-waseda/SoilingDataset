/****************************************************************************
** Form interface generated from reading ui file 'Qt/BeoSubSimuQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef BEOSUBSIMUQTMAINFORM_H
#define BEOSUBSIMUQTMAINFORM_H

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
class Simulation;

class BeoSubSimuQtMainForm : public QWidget
{
    Q_OBJECT

public:
    BeoSubSimuQtMainForm( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~BeoSubSimuQtMainForm();

    Simulation* simulation1;

    void init();

protected:

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // BEOSUBSIMUQTMAINFORM_H
