/****************************************************************************
** Form interface generated from reading ui file 'Qt/BeoSubQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef BEOSUBQT_H
#define BEOSUBQT_H

#include <qvariant.h>
#include <qdialog.h>
#include "Component/ModelManager.H"
#include "BeoSub/BeoSubOneBal.H"
#include "Util/StringConversions.H"
#include "Util/Angle.H"
#include "Devices/JoyStick.H"
#include "Parallel/pvisionTCP-defs.H"
#include "Beowulf/Beowulf.H"
#include "BeoSub/ColorTracker.H"
#include "BeoSub/BeoSubTaskDecoder.H"
#include "BeoSub/BeoSubCanny.H"
#include "BeoSub/CannyModel.H"
#include "BeoSub/BeoSubDB.H"
#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectDB.H"
#include "SIFT/Keypoint.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QLabel;
class QPushButton;
class QScrollBar;
class QRadioButton;
class QLineEdit;

class BeoSubQtMainForm : public QDialog
{
    Q_OBJECT

public:
    BeoSubQtMainForm( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~BeoSubQtMainForm();

    QLabel* bottomfilename;
    QPushButton* saveDown;
    QLabel* frontfilename;
    QPushButton* saveFront;
    QPushButton* saveUp;
    QLabel* topfilename;
    QLabel* textLabel2_2_2_2;
    QLabel* TsDisplay;
    QScrollBar* DiveScroll;
    QScrollBar* FrontBScroll;
    QLabel* textLabel2_2_2;
    QLabel* FrontBDisplay;
    QLabel* LineEdit12;
    QScrollBar* TsScroll;
    QLabel* BsDisplay;
    QLabel* textLabel2_2_2_3;
    QScrollBar* RearBScroll;
    QScrollBar* OrientScroll;
    QLabel* LTDisplay;
    QLabel* textLabel2_2;
    QLabel* LineEdit16;
    QScrollBar* RTScroll;
    QScrollBar* LTScroll;
    QLabel* TiltDisplay;
    QLabel* RTDisplay;
    QLabel* RearBDisplay;
    QLabel* TurnDisplay;
    QLabel* Label_2;
    QLabel* textLabel3;
    QLabel* textLabel2;
    QLabel* textLabel3_2;
    QLabel* textLabel1;
    QLabel* ImagePixmapLabel1;
    QLabel* LineEdit2_2;
    QLabel* LineEdit1;
    QLabel* LineEdit2;
    QLabel* LineEdit3_2;
    QLabel* LineEdit4;
    QLabel* LineEdit3;
    QLabel* LineEdit1_2;
    QLabel* LineEdit4_2;
    QLabel* Label_1_2;
    QLabel* Label_1_2_2_2;
    QLabel* Label_1_2_2;
    QLabel* LineEdit11;
    QLabel* textLabel11;
    QLabel* LineEdit15;
    QLabel* textLabel15;
    QLabel* textLabel16;
    QLabel* textLabel12;
    QScrollBar* PitchScroll;
    QRadioButton* PitchPID;
    QRadioButton* HeadingPID;
    QRadioButton* KILL;
    QRadioButton* DepthPID;
    QLabel* LineEdit11_2_2;
    QPushButton* ButtonStop;
    QLabel* ImagePixmapLabel2;
    QLabel* LabelTaskControl_2;
    QLabel* CPUtemp;
    QLineEdit* lineEditAdvance;
    QLineEdit* lineEditStrafe;
    QPushButton* resetA;
    QPushButton* GetDirections;
    QLabel* textLabel12_2_2;
    QPushButton* ButtonTaskB;
    QPushButton* ButtonDecode;
    QPushButton* ButtonTaskC;
    QLabel* LabelTaskControl;
    QPushButton* ButtonTaskA;
    QLineEdit* DepthD;
    QLineEdit* DepthP;
    QLabel* LineEdit11_2_3;
    QLineEdit* PitchD;
    QLineEdit* DepthI;
    QLineEdit* PitchI;
    QLineEdit* PitchP;
    QLabel* LineEdit11_2_3_2;
    QLineEdit* HeadingI;
    QLineEdit* HeadingP;
    QLabel* LineEdit11_2_3_3;
    QLineEdit* HeadingD;
    QLabel* LineEdit11_2;
    QLabel* LineEdit11_2_2_2;
    QLabel* LineEdit11_2_2_2_2;
    QLabel* LabelTaskControl_2_2;
    QLabel* indicator;
    QLabel* textLabel12_2;
    QLabel* textLabel12_2_3;
    QPushButton* ButtonGate;
    QPushButton* ButtonPic;
    QLineEdit* to;
    QLabel* ImagePixmapLabel3;

    bool joyflag;
    nub::soft_ref<Beowulf> itsBeo;
    TCPmessage smsg;
    TCPmessage rmsg;
    int counterDown;
    int counterFront;
    int counterUp;
    nub::soft_ref<BeoSubOneBal> itsSub;
    Image< PixRGB<byte> > imup;
    Image< PixRGB<byte> > imfront;
    Image< PixRGB<byte> > imdown;
    nub::soft_ref<BeoSubTaskDecoder> itsDecoder;
    nub::soft_ref<ColorTracker> itsTracker1;
    nub::soft_ref<ColorTracker> itsTracker2;
    nub::soft_ref<BeoSubCanny> itsDetector;
    bool takePic;

public slots:
    virtual void init( const nub::soft_ref<BeoSubOneBal> & sub, const nub::soft_ref<Beowulf> & beo, const nub::soft_ref<BeoSubTaskDecoder> & decoder, const nub::soft_ref<ColorTracker> & tracker1, const nub::soft_ref<ColorTracker> & tracker2, const nub::soft_ref<BeoSubCanny> & detector );
    virtual void togglePic();
    virtual void savePIDvalue();
    virtual void HeadingPID_toggled();
    virtual void PitchPID_toggled();
    virtual void DepthPID_toggled();
    virtual void KILL_toggled();
    virtual void turnopen_return();
    virtual void advance_return();
    virtual void strafe_return();
    virtual void resetAtt_return();
    virtual void FrontBallast_valueChanged( int val );
    virtual void RearBallast_valueChanged( int val );
    virtual void LThruster_valueChanged( int val );
    virtual void RThuster_valueChanged( int val );
    virtual void Thrusters_valueChanged( int val );
    virtual void dive_valueChanged( int val );
    virtual void Orient_valueChanged( int val );
    virtual void Pitch_valueChanged( int val );
    virtual void timerEvent( QTimerEvent * e );
    virtual void saveImageLog( const char * name, int count );
    virtual void saveImageUp();
    virtual void saveImageFront();
    virtual void saveImageDown();
    virtual void taskGate();
    virtual void taskDecode();
    virtual void taskA();
    virtual void taskB();
    virtual void taskC();
    virtual void stopAll();
    virtual void matchAndDirect();
    virtual void parseMessage( TCPmessage & rmsg );
    virtual void displayFunc();
    virtual void keyPressEvent( QKeyEvent * event );
    virtual void keyReleaseEvent( QKeyEvent * event );

protected:
    int newVariable;
    ModelManager manager;
    pthread_t IMU_thread;
    std::list<std::string> topFiles;
    std::list<std::string> frontFiles;
    std::list<std::string> bottomFiles;


protected slots:
    virtual void languageChange();

private:
    FILE *savePIDgain;
    uint i;

};

#endif // BEOSUBQT_H
