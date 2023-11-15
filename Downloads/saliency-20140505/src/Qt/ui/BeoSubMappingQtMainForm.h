/****************************************************************************
** Form interface generated from reading ui file 'Qt/BeoSubMappingQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef BEOSUBMAPPINGQTMAINFORM_H
#define BEOSUBMAPPINGQTMAINFORM_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qdialog.h>
#include "Image/Image.H"
#include "Component/ModelManager.H"
#include "Util/StringConversions.H"
#include "Parallel/pvisionTCP-defs.H"
#include "Beowulf/Beowulf.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class PoolImage;
class QLabel;
class QPushButton;
class QListBox;
class QListBoxItem;
class QLineEdit;

class BeoSubMappingQtMainForm : public QDialog
{
    Q_OBJECT

public:
    BeoSubMappingQtMainForm( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~BeoSubMappingQtMainForm();

    QLabel* LabelLog;
    QLabel* Log;
    QPushButton* save;
    QListBox* List;
    QLabel* LabelImage;
    PoolImage* poolImage;
    QLabel* displayCoord;
    QLabel* LabelList;
    QPushButton* loadAS;
    QPushButton* load;
    QPushButton* restHeading;
    QLabel* test;
    QLabel* displayImage;
    QLineEdit* changeHeading;
    QLabel* LabelImage_2;
    QPushButton* refresh;
    QPushButton* change2bottom;
    QPushButton* change2front;
    QPushButton* change2top;

    Image< PixRGB<byte> > img;
    bool setSave;
    std::string prefix;

public slots:
    virtual void init();
    virtual void change_bottom();
    virtual void change_front();
    virtual void change_top();
    virtual void camera_return();
    virtual void displayFunc();
    virtual void timerEvent( QTimerEvent * e );
    virtual void displayLog();
    virtual void loadList();
    virtual void refreshImages();
    virtual void createIcon();
    virtual void saveiconList();
    virtual void LoadAngleScale();
    virtual void resetAllHeading();
    virtual void changeTheHeading();

protected:

protected slots:
    virtual void languageChange();

private:
    int numItem;

    QPixmap image0;

};

#endif // BEOSUBMAPPINGQTMAINFORM_H
