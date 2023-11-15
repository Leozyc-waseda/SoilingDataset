/****************************************************************************
** Form interface generated from reading ui file 'Qt/RobotHeadForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef ROBOTHEADFORM_H
#define ROBOTHEADFORM_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qmainwindow.h>
#include <qthread.h>
#include "Component/ModelManager.H"
#include "Devices/BeoHead.H"
#include "Image/Image.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class QTabWidget;
class QWidget;
class QGroupBox;
class QLabel;
class QLineEdit;
class QSlider;
class QPushButton;

class RobotHeadForm : public QMainWindow
{
    Q_OBJECT

public:
    RobotHeadForm( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~RobotHeadForm();

    QTabWidget* tabWidget2;
    QWidget* tab;
    QGroupBox* groupBox1;
    QLabel* textLabel1;
    QLineEdit* leftEyePanDisp;
    QSlider* leftEyeTiltSlide;
    QLineEdit* leftEyeTiltDisp;
    QLabel* textLabel2;
    QSlider* leftEyePanSlide;
    QGroupBox* groupBox1_2;
    QLabel* textLabel1_2;
    QLineEdit* rightEyePanDisp;
    QSlider* rightEyeTiltSlide;
    QLineEdit* rightEyeTiltDisp;
    QLabel* textLabel2_2;
    QSlider* rightEyePanSlide;
    QGroupBox* groupBox1_3;
    QLabel* textLabel1_3;
    QLineEdit* headPanDisp;
    QSlider* headPanSlide;
    QLabel* textLabel2_3;
    QSlider* headTiltSlide;
    QLineEdit* headTiltDisp;
    QLineEdit* headYawDisp;
    QSlider* headYawSlide;
    QLabel* textLabel2_3_2;
    QPushButton* restPosButton;
    QPushButton* relaxNeckButton;
    QWidget* tab_2;
    QMenuBar *MenuBar;
    QPopupMenu *fileMenu;
    QPopupMenu *editMenu;
    QPopupMenu *helpMenu;
    QAction* fileNewAction;
    QAction* fileOpenAction;
    QAction* fileSaveAction;
    QAction* fileSaveAsAction;
    QAction* filePrintAction;
    QAction* fileExitAction;
    QAction* editUndoAction;
    QAction* editRedoAction;
    QAction* editCutAction;
    QAction* editCopyAction;
    QAction* editPasteAction;
    QAction* editFindAction;
    QAction* helpContentsAction;
    QAction* helpIndexAction;
    QAction* helpAboutAction;

    virtual void init( ModelManager & mgr, nub::soft_ref<BeoHead> beoHead );

public slots:
    virtual void fileNew();
    virtual void fileOpen();
    virtual void fileSave();
    virtual void fileSaveAs();
    virtual void filePrint();
    virtual void fileExit();
    virtual void editUndo();
    virtual void editRedo();
    virtual void editCut();
    virtual void editPaste();
    virtual void editFind();
    virtual void helpIndex();
    virtual void helpContents();
    virtual void helpAbout();
    virtual void grab();
    virtual void moveLeftEyePan( int pos );
    virtual void moveLeftEyeTilt( int pos );
    virtual void moveRightEyePan( int pos );
    virtual void moveRightEyeTilt( int pos );
    virtual void moveHeadPan( int pos );
    virtual void moveHeadTilt( int pos );
    virtual void moveHeadYaw( int pos );
    virtual void relaxNeck();
    virtual void restPos();

protected:
    ModelManager *itsMgr;
    nub::soft_ref<BeoHead> itsBeoHead;

    QVBoxLayout* RobotHeadFormLayout;
    QVBoxLayout* tabLayout;
    QHBoxLayout* layout7;
    QGridLayout* groupBox1Layout;
    QGridLayout* groupBox1_2Layout;
    QGridLayout* groupBox1_3Layout;
    QHBoxLayout* layout3;
    QSpacerItem* spacer1;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;
    QPixmap image1;
    QPixmap image2;
    QPixmap image3;
    QPixmap image4;
    QPixmap image5;
    QPixmap image6;
    QPixmap image7;
    QPixmap image8;
    QPixmap image9;

};

#endif // ROBOTHEADFORM_H
