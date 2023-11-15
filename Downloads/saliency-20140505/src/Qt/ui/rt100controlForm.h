/****************************************************************************
** Form interface generated from reading ui file 'Qt/rt100controlForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef RT100CONTROLFORM_H
#define RT100CONTROLFORM_H

#include <qvariant.h>
#include <qdialog.h>
#include "Component/ModelManager.H"
#include "Devices/rt100.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QGroupBox;
class QLabel;
class QSpinBox;
class QCheckBox;
class QPushButton;
class QRadioButton;
class QLineEdit;

class RT100ControlForm : public QDialog
{
    Q_OBJECT

public:
    RT100ControlForm( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~RT100ControlForm();

    QGroupBox* groupBox1;
    QLabel* textLabel1;
    QSpinBox* zedVal;
    QLabel* textLabel1_2;
    QSpinBox* sholderVal;
    QLabel* textLabel1_3;
    QSpinBox* elbowVal;
    QLabel* textLabel1_4;
    QSpinBox* yawVal;
    QLabel* textLabel1_5;
    QSpinBox* wrist1Val;
    QLabel* textLabel1_6;
    QSpinBox* wrist2Val;
    QLabel* textLabel1_6_2;
    QSpinBox* gripperVal;
    QLabel* textLabel1_7;
    QSpinBox* wristTiltVal;
    QLabel* textLabel2_2;
    QSpinBox* wristRollVal;
    QCheckBox* moveMode;
    QPushButton* moveArmButton;
    QPushButton* getJointsPositionButton;
    QGroupBox* groupBox2;
    QLabel* textLabel1_8;
    QSpinBox* zedInterpolation;
    QLabel* textLabel1_2_3;
    QSpinBox* sholderInterpolation;
    QLabel* textLabel1_3_3;
    QSpinBox* elbowInterpolation;
    QLabel* textLabel1_4_3;
    QSpinBox* yawInterpolation;
    QLabel* textLabel1_5_3;
    QSpinBox* wrist1Interpolation;
    QLabel* textLabel1_6_4;
    QSpinBox* wrist2Interpolation;
    QLabel* textLabel1_6_2_3;
    QSpinBox* gripperInterpolation;
    QLabel* textLabel1_7_3;
    QSpinBox* wristTiltInterpolation;
    QLabel* textLabel2_2_3;
    QSpinBox* wristRollInterpolation;
    QPushButton* doInterpolationButton;
    QGroupBox* groupBox7;
    QLabel* textLabel2_3;
    QLabel* textLabel2_3_2;
    QLabel* textLabel2_3_3;
    QLabel* textLabel2_3_4;
    QLabel* textLabel1_9;
    QRadioButton* radioButton40;
    QRadioButton* radioButton40_2;
    QRadioButton* radioButton40_3;
    QRadioButton* radioButton40_4;
    QLabel* textLabel1_9_2;
    QRadioButton* radioButton40_5;
    QRadioButton* radioButton40_2_2;
    QRadioButton* radioButton40_3_2;
    QRadioButton* radioButton40_4_2;
    QLabel* textLabel1_9_3;
    QRadioButton* radioButton40_6;
    QRadioButton* radioButton40_2_3;
    QRadioButton* radioButton40_3_3;
    QRadioButton* radioButton40_4_3;
    QLabel* textLabel1_9_4;
    QRadioButton* radioButton40_7;
    QRadioButton* radioButton40_2_4;
    QRadioButton* radioButton40_3_4;
    QRadioButton* radioButton40_4_4;
    QLabel* textLabel1_9_5;
    QRadioButton* radioButton40_8;
    QRadioButton* radioButton40_2_5;
    QRadioButton* radioButton40_3_5;
    QRadioButton* radioButton40_4_5;
    QLabel* textLabel1_9_6;
    QRadioButton* radioButton40_9;
    QRadioButton* radioButton40_2_6;
    QRadioButton* radioButton40_3_6;
    QRadioButton* radioButton40_4_6;
    QLabel* textLabel1_9_7;
    QRadioButton* radioButton40_10;
    QRadioButton* radioButton40_2_7;
    QRadioButton* radioButton40_3_7;
    QRadioButton* radioButton40_4_7;
    QPushButton* manualMoveBtn;
    QPushButton* allStopBtn;
    QPushButton* initArmButton;
    QPushButton* homeArmButton;
    QPushButton* quitButton;
    QLabel* textLabel2;
    QLabel* armStatusMsg;
    QGroupBox* groupBox4;
    QLineEdit* parErr;
    QLineEdit* parCuPos;
    QLineEdit* parErrLimit;
    QLineEdit* parNewPos;
    QLineEdit* parSpeed;
    QLineEdit* parKP;
    QLineEdit* parKI;
    QLineEdit* parKD;
    QLineEdit* parDeadBand;
    QLineEdit* parOffset;
    QLineEdit* parMaxForce;
    QLineEdit* parCurrForce;
    QLineEdit* parAccTime;
    QLineEdit* parUserIO;
    QLabel* textLabel3;
    QLabel* textLabel3_2;
    QLabel* textLabel3_3;
    QLabel* textLabel3_4;
    QLabel* textLabel3_5;
    QLabel* textLabel3_6;
    QLabel* textLabel3_7;
    QLabel* textLabel3_8;
    QLabel* textLabel3_9;
    QLabel* textLabel1_10_3;
    QLabel* textLabel1_10_2;
    QLabel* textLabel1_10;
    QLabel* textLabel1_10_4;
    QLabel* textLabel1_10_5;

    virtual void init( ModelManager * mgr, nub::soft_ref<RT100> rt100 );

public slots:
    virtual void initArm();
    virtual void homeArm();
    virtual void moveArm();
    virtual void moveZed( int val );
    virtual void moveSholder( int val );
    virtual void moveElbow( int val );
    virtual void moveYaw( int val );
    virtual void moveWrist1( int val );
    virtual void moveWrist2( int val );
    virtual void moveGripper( int val );
    virtual void getCurrentJointPositions();
    virtual void wristRoll( int val );
    virtual void wristTilt( int val );
    virtual void doInterpolation();

protected:
    nub::soft_ref<RT100> itsRT100;
    ModelManager* itsMgr;

    QVBoxLayout* groupBox1Layout;
    QHBoxLayout* layout5;
    QHBoxLayout* layout6;
    QHBoxLayout* layout7;
    QHBoxLayout* layout8;
    QHBoxLayout* layout9;
    QHBoxLayout* layout10;
    QHBoxLayout* layout11;
    QHBoxLayout* layout12;
    QHBoxLayout* layout11_2;
    QVBoxLayout* groupBox2Layout;
    QHBoxLayout* layout22;
    QHBoxLayout* layout23;
    QHBoxLayout* layout24;
    QHBoxLayout* layout25;
    QHBoxLayout* layout26;
    QHBoxLayout* layout27;
    QHBoxLayout* layout28;
    QHBoxLayout* layout29;
    QHBoxLayout* layout30;
    QVBoxLayout* groupBox7Layout;
    QHBoxLayout* layout34;
    QSpacerItem* spacer3;
    QHBoxLayout* layout33;
    QHBoxLayout* layout35;
    QHBoxLayout* layout36;
    QHBoxLayout* layout37;
    QHBoxLayout* layout38;
    QHBoxLayout* layout39;
    QHBoxLayout* layout40;
    QHBoxLayout* layout41;
    QSpacerItem* spacer4;
    QHBoxLayout* layout11_3;
    QSpacerItem* spacer7;
    QHBoxLayout* layout12_2;
    QHBoxLayout* layout32;
    QHBoxLayout* layout31;

protected slots:
    virtual void languageChange();

};

#endif // RT100CONTROLFORM_H
