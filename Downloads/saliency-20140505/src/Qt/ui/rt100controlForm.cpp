/****************************************************************************
** Form implementation generated from reading ui file 'Qt/rt100controlForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/rt100controlForm.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qspinbox.h>
#include <qcheckbox.h>
#include <qradiobutton.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Qt/rt100controlForm.ui.h"

/*
 *  Constructs a RT100ControlForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
RT100ControlForm::RT100ControlForm( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "RT100ControlForm" );

    groupBox1 = new QGroupBox( this, "groupBox1" );
    groupBox1->setGeometry( QRect( 12, 12, 124, 410 ) );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 11 );
    groupBox1Layout = new QVBoxLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    layout5 = new QHBoxLayout( 0, 0, 6, "layout5");

    textLabel1 = new QLabel( groupBox1, "textLabel1" );
    layout5->addWidget( textLabel1 );

    zedVal = new QSpinBox( groupBox1, "zedVal" );
    zedVal->setMaxValue( 0 );
    zedVal->setMinValue( -3300 );
    zedVal->setLineStep( 100 );
    layout5->addWidget( zedVal );
    groupBox1Layout->addLayout( layout5 );

    layout6 = new QHBoxLayout( 0, 0, 6, "layout6");

    textLabel1_2 = new QLabel( groupBox1, "textLabel1_2" );
    layout6->addWidget( textLabel1_2 );

    sholderVal = new QSpinBox( groupBox1, "sholderVal" );
    sholderVal->setMaxValue( 2663 );
    sholderVal->setMinValue( -2773 );
    sholderVal->setLineStep( 100 );
    sholderVal->setValue( 0 );
    layout6->addWidget( sholderVal );
    groupBox1Layout->addLayout( layout6 );

    layout7 = new QHBoxLayout( 0, 0, 6, "layout7");

    textLabel1_3 = new QLabel( groupBox1, "textLabel1_3" );
    layout7->addWidget( textLabel1_3 );

    elbowVal = new QSpinBox( groupBox1, "elbowVal" );
    elbowVal->setMaxValue( 2500 );
    elbowVal->setMinValue( -2900 );
    elbowVal->setLineStep( 100 );
    elbowVal->setValue( 0 );
    layout7->addWidget( elbowVal );
    groupBox1Layout->addLayout( layout7 );

    layout8 = new QHBoxLayout( 0, 0, 6, "layout8");

    textLabel1_4 = new QLabel( groupBox1, "textLabel1_4" );
    layout8->addWidget( textLabel1_4 );

    yawVal = new QSpinBox( groupBox1, "yawVal" );
    yawVal->setMaxValue( 1088 );
    yawVal->setMinValue( -1104 );
    yawVal->setLineStep( 100 );
    yawVal->setValue( 0 );
    layout8->addWidget( yawVal );
    groupBox1Layout->addLayout( layout8 );

    layout9 = new QHBoxLayout( 0, 0, 6, "layout9");

    textLabel1_5 = new QLabel( groupBox1, "textLabel1_5" );
    layout9->addWidget( textLabel1_5 );

    wrist1Val = new QSpinBox( groupBox1, "wrist1Val" );
    wrist1Val->setMaxValue( 4000 );
    wrist1Val->setMinValue( -4000 );
    wrist1Val->setLineStep( 100 );
    layout9->addWidget( wrist1Val );
    groupBox1Layout->addLayout( layout9 );

    layout10 = new QHBoxLayout( 0, 0, 6, "layout10");

    textLabel1_6 = new QLabel( groupBox1, "textLabel1_6" );
    layout10->addWidget( textLabel1_6 );

    wrist2Val = new QSpinBox( groupBox1, "wrist2Val" );
    wrist2Val->setMaxValue( 4000 );
    wrist2Val->setMinValue( -4000 );
    wrist2Val->setLineStep( 100 );
    layout10->addWidget( wrist2Val );
    groupBox1Layout->addLayout( layout10 );

    layout11 = new QHBoxLayout( 0, 0, 6, "layout11");

    textLabel1_6_2 = new QLabel( groupBox1, "textLabel1_6_2" );
    layout11->addWidget( textLabel1_6_2 );

    gripperVal = new QSpinBox( groupBox1, "gripperVal" );
    gripperVal->setMaxValue( 1183 );
    gripperVal->setMinValue( -500 );
    gripperVal->setLineStep( 100 );
    gripperVal->setValue( 0 );
    layout11->addWidget( gripperVal );
    groupBox1Layout->addLayout( layout11 );

    layout12 = new QHBoxLayout( 0, 0, 6, "layout12");

    textLabel1_7 = new QLabel( groupBox1, "textLabel1_7" );
    layout12->addWidget( textLabel1_7 );

    wristTiltVal = new QSpinBox( groupBox1, "wristTiltVal" );
    wristTiltVal->setMaxValue( 4000 );
    wristTiltVal->setMinValue( -4000 );
    wristTiltVal->setLineStep( 100 );
    layout12->addWidget( wristTiltVal );
    groupBox1Layout->addLayout( layout12 );

    layout11_2 = new QHBoxLayout( 0, 0, 6, "layout11_2");

    textLabel2_2 = new QLabel( groupBox1, "textLabel2_2" );
    layout11_2->addWidget( textLabel2_2 );

    wristRollVal = new QSpinBox( groupBox1, "wristRollVal" );
    wristRollVal->setMaxValue( 4000 );
    wristRollVal->setMinValue( -4000 );
    wristRollVal->setLineStep( 100 );
    layout11_2->addWidget( wristRollVal );
    groupBox1Layout->addLayout( layout11_2 );

    moveMode = new QCheckBox( groupBox1, "moveMode" );
    groupBox1Layout->addWidget( moveMode );

    moveArmButton = new QPushButton( groupBox1, "moveArmButton" );
    groupBox1Layout->addWidget( moveArmButton );

    getJointsPositionButton = new QPushButton( groupBox1, "getJointsPositionButton" );
    groupBox1Layout->addWidget( getJointsPositionButton );

    groupBox2 = new QGroupBox( this, "groupBox2" );
    groupBox2->setGeometry( QRect( 140, 10, 124, 410 ) );
    groupBox2->setColumnLayout(0, Qt::Vertical );
    groupBox2->layout()->setSpacing( 6 );
    groupBox2->layout()->setMargin( 11 );
    groupBox2Layout = new QVBoxLayout( groupBox2->layout() );
    groupBox2Layout->setAlignment( Qt::AlignTop );

    layout22 = new QHBoxLayout( 0, 0, 6, "layout22");

    textLabel1_8 = new QLabel( groupBox2, "textLabel1_8" );
    layout22->addWidget( textLabel1_8 );

    zedInterpolation = new QSpinBox( groupBox2, "zedInterpolation" );
    zedInterpolation->setMaxValue( 7 );
    zedInterpolation->setMinValue( -8 );
    layout22->addWidget( zedInterpolation );
    groupBox2Layout->addLayout( layout22 );

    layout23 = new QHBoxLayout( 0, 0, 6, "layout23");

    textLabel1_2_3 = new QLabel( groupBox2, "textLabel1_2_3" );
    layout23->addWidget( textLabel1_2_3 );

    sholderInterpolation = new QSpinBox( groupBox2, "sholderInterpolation" );
    sholderInterpolation->setMaxValue( 7 );
    sholderInterpolation->setMinValue( -8 );
    layout23->addWidget( sholderInterpolation );
    groupBox2Layout->addLayout( layout23 );

    layout24 = new QHBoxLayout( 0, 0, 6, "layout24");

    textLabel1_3_3 = new QLabel( groupBox2, "textLabel1_3_3" );
    layout24->addWidget( textLabel1_3_3 );

    elbowInterpolation = new QSpinBox( groupBox2, "elbowInterpolation" );
    elbowInterpolation->setMaxValue( 7 );
    elbowInterpolation->setMinValue( -8 );
    layout24->addWidget( elbowInterpolation );
    groupBox2Layout->addLayout( layout24 );

    layout25 = new QHBoxLayout( 0, 0, 6, "layout25");

    textLabel1_4_3 = new QLabel( groupBox2, "textLabel1_4_3" );
    layout25->addWidget( textLabel1_4_3 );

    yawInterpolation = new QSpinBox( groupBox2, "yawInterpolation" );
    yawInterpolation->setMaxValue( 7 );
    yawInterpolation->setMinValue( -8 );
    layout25->addWidget( yawInterpolation );
    groupBox2Layout->addLayout( layout25 );

    layout26 = new QHBoxLayout( 0, 0, 6, "layout26");

    textLabel1_5_3 = new QLabel( groupBox2, "textLabel1_5_3" );
    layout26->addWidget( textLabel1_5_3 );

    wrist1Interpolation = new QSpinBox( groupBox2, "wrist1Interpolation" );
    wrist1Interpolation->setMaxValue( 7 );
    wrist1Interpolation->setMinValue( -8 );
    layout26->addWidget( wrist1Interpolation );
    groupBox2Layout->addLayout( layout26 );

    layout27 = new QHBoxLayout( 0, 0, 6, "layout27");

    textLabel1_6_4 = new QLabel( groupBox2, "textLabel1_6_4" );
    layout27->addWidget( textLabel1_6_4 );

    wrist2Interpolation = new QSpinBox( groupBox2, "wrist2Interpolation" );
    wrist2Interpolation->setMaxValue( 7 );
    wrist2Interpolation->setMinValue( -8 );
    layout27->addWidget( wrist2Interpolation );
    groupBox2Layout->addLayout( layout27 );

    layout28 = new QHBoxLayout( 0, 0, 6, "layout28");

    textLabel1_6_2_3 = new QLabel( groupBox2, "textLabel1_6_2_3" );
    layout28->addWidget( textLabel1_6_2_3 );

    gripperInterpolation = new QSpinBox( groupBox2, "gripperInterpolation" );
    gripperInterpolation->setMaxValue( 7 );
    gripperInterpolation->setMinValue( -8 );
    layout28->addWidget( gripperInterpolation );
    groupBox2Layout->addLayout( layout28 );

    layout29 = new QHBoxLayout( 0, 0, 6, "layout29");

    textLabel1_7_3 = new QLabel( groupBox2, "textLabel1_7_3" );
    layout29->addWidget( textLabel1_7_3 );

    wristTiltInterpolation = new QSpinBox( groupBox2, "wristTiltInterpolation" );
    wristTiltInterpolation->setMaxValue( 7 );
    wristTiltInterpolation->setMinValue( -8 );
    layout29->addWidget( wristTiltInterpolation );
    groupBox2Layout->addLayout( layout29 );

    layout30 = new QHBoxLayout( 0, 0, 6, "layout30");

    textLabel2_2_3 = new QLabel( groupBox2, "textLabel2_2_3" );
    layout30->addWidget( textLabel2_2_3 );

    wristRollInterpolation = new QSpinBox( groupBox2, "wristRollInterpolation" );
    wristRollInterpolation->setMaxValue( 7 );
    wristRollInterpolation->setMinValue( -8 );
    layout30->addWidget( wristRollInterpolation );
    groupBox2Layout->addLayout( layout30 );

    doInterpolationButton = new QPushButton( groupBox2, "doInterpolationButton" );
    groupBox2Layout->addWidget( doInterpolationButton );

    groupBox7 = new QGroupBox( this, "groupBox7" );
    groupBox7->setGeometry( QRect( 270, 10, 267, 297 ) );
    groupBox7->setColumnLayout(0, Qt::Vertical );
    groupBox7->layout()->setSpacing( 6 );
    groupBox7->layout()->setMargin( 11 );
    groupBox7Layout = new QVBoxLayout( groupBox7->layout() );
    groupBox7Layout->setAlignment( Qt::AlignTop );

    layout34 = new QHBoxLayout( 0, 0, 6, "layout34");
    spacer3 = new QSpacerItem( 20, 21, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout34->addItem( spacer3 );

    textLabel2_3 = new QLabel( groupBox7, "textLabel2_3" );
    layout34->addWidget( textLabel2_3 );

    textLabel2_3_2 = new QLabel( groupBox7, "textLabel2_3_2" );
    layout34->addWidget( textLabel2_3_2 );

    textLabel2_3_3 = new QLabel( groupBox7, "textLabel2_3_3" );
    layout34->addWidget( textLabel2_3_3 );

    textLabel2_3_4 = new QLabel( groupBox7, "textLabel2_3_4" );
    layout34->addWidget( textLabel2_3_4 );
    groupBox7Layout->addLayout( layout34 );

    layout33 = new QHBoxLayout( 0, 0, 6, "layout33");

    textLabel1_9 = new QLabel( groupBox7, "textLabel1_9" );
    layout33->addWidget( textLabel1_9 );

    radioButton40 = new QRadioButton( groupBox7, "radioButton40" );
    layout33->addWidget( radioButton40 );

    radioButton40_2 = new QRadioButton( groupBox7, "radioButton40_2" );
    layout33->addWidget( radioButton40_2 );

    radioButton40_3 = new QRadioButton( groupBox7, "radioButton40_3" );
    layout33->addWidget( radioButton40_3 );

    radioButton40_4 = new QRadioButton( groupBox7, "radioButton40_4" );
    layout33->addWidget( radioButton40_4 );
    groupBox7Layout->addLayout( layout33 );

    layout35 = new QHBoxLayout( 0, 0, 6, "layout35");

    textLabel1_9_2 = new QLabel( groupBox7, "textLabel1_9_2" );
    layout35->addWidget( textLabel1_9_2 );

    radioButton40_5 = new QRadioButton( groupBox7, "radioButton40_5" );
    layout35->addWidget( radioButton40_5 );

    radioButton40_2_2 = new QRadioButton( groupBox7, "radioButton40_2_2" );
    layout35->addWidget( radioButton40_2_2 );

    radioButton40_3_2 = new QRadioButton( groupBox7, "radioButton40_3_2" );
    layout35->addWidget( radioButton40_3_2 );

    radioButton40_4_2 = new QRadioButton( groupBox7, "radioButton40_4_2" );
    layout35->addWidget( radioButton40_4_2 );
    groupBox7Layout->addLayout( layout35 );

    layout36 = new QHBoxLayout( 0, 0, 6, "layout36");

    textLabel1_9_3 = new QLabel( groupBox7, "textLabel1_9_3" );
    layout36->addWidget( textLabel1_9_3 );

    radioButton40_6 = new QRadioButton( groupBox7, "radioButton40_6" );
    layout36->addWidget( radioButton40_6 );

    radioButton40_2_3 = new QRadioButton( groupBox7, "radioButton40_2_3" );
    layout36->addWidget( radioButton40_2_3 );

    radioButton40_3_3 = new QRadioButton( groupBox7, "radioButton40_3_3" );
    layout36->addWidget( radioButton40_3_3 );

    radioButton40_4_3 = new QRadioButton( groupBox7, "radioButton40_4_3" );
    layout36->addWidget( radioButton40_4_3 );
    groupBox7Layout->addLayout( layout36 );

    layout37 = new QHBoxLayout( 0, 0, 6, "layout37");

    textLabel1_9_4 = new QLabel( groupBox7, "textLabel1_9_4" );
    layout37->addWidget( textLabel1_9_4 );

    radioButton40_7 = new QRadioButton( groupBox7, "radioButton40_7" );
    layout37->addWidget( radioButton40_7 );

    radioButton40_2_4 = new QRadioButton( groupBox7, "radioButton40_2_4" );
    layout37->addWidget( radioButton40_2_4 );

    radioButton40_3_4 = new QRadioButton( groupBox7, "radioButton40_3_4" );
    layout37->addWidget( radioButton40_3_4 );

    radioButton40_4_4 = new QRadioButton( groupBox7, "radioButton40_4_4" );
    layout37->addWidget( radioButton40_4_4 );
    groupBox7Layout->addLayout( layout37 );

    layout38 = new QHBoxLayout( 0, 0, 6, "layout38");

    textLabel1_9_5 = new QLabel( groupBox7, "textLabel1_9_5" );
    layout38->addWidget( textLabel1_9_5 );

    radioButton40_8 = new QRadioButton( groupBox7, "radioButton40_8" );
    layout38->addWidget( radioButton40_8 );

    radioButton40_2_5 = new QRadioButton( groupBox7, "radioButton40_2_5" );
    layout38->addWidget( radioButton40_2_5 );

    radioButton40_3_5 = new QRadioButton( groupBox7, "radioButton40_3_5" );
    layout38->addWidget( radioButton40_3_5 );

    radioButton40_4_5 = new QRadioButton( groupBox7, "radioButton40_4_5" );
    layout38->addWidget( radioButton40_4_5 );
    groupBox7Layout->addLayout( layout38 );

    layout39 = new QHBoxLayout( 0, 0, 6, "layout39");

    textLabel1_9_6 = new QLabel( groupBox7, "textLabel1_9_6" );
    layout39->addWidget( textLabel1_9_6 );

    radioButton40_9 = new QRadioButton( groupBox7, "radioButton40_9" );
    layout39->addWidget( radioButton40_9 );

    radioButton40_2_6 = new QRadioButton( groupBox7, "radioButton40_2_6" );
    layout39->addWidget( radioButton40_2_6 );

    radioButton40_3_6 = new QRadioButton( groupBox7, "radioButton40_3_6" );
    layout39->addWidget( radioButton40_3_6 );

    radioButton40_4_6 = new QRadioButton( groupBox7, "radioButton40_4_6" );
    layout39->addWidget( radioButton40_4_6 );
    groupBox7Layout->addLayout( layout39 );

    layout40 = new QHBoxLayout( 0, 0, 6, "layout40");

    textLabel1_9_7 = new QLabel( groupBox7, "textLabel1_9_7" );
    layout40->addWidget( textLabel1_9_7 );

    radioButton40_10 = new QRadioButton( groupBox7, "radioButton40_10" );
    layout40->addWidget( radioButton40_10 );

    radioButton40_2_7 = new QRadioButton( groupBox7, "radioButton40_2_7" );
    layout40->addWidget( radioButton40_2_7 );

    radioButton40_3_7 = new QRadioButton( groupBox7, "radioButton40_3_7" );
    layout40->addWidget( radioButton40_3_7 );

    radioButton40_4_7 = new QRadioButton( groupBox7, "radioButton40_4_7" );
    layout40->addWidget( radioButton40_4_7 );
    groupBox7Layout->addLayout( layout40 );

    layout41 = new QHBoxLayout( 0, 0, 6, "layout41");

    manualMoveBtn = new QPushButton( groupBox7, "manualMoveBtn" );
    layout41->addWidget( manualMoveBtn );

    allStopBtn = new QPushButton( groupBox7, "allStopBtn" );
    layout41->addWidget( allStopBtn );
    spacer4 = new QSpacerItem( 71, 21, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout41->addItem( spacer4 );
    groupBox7Layout->addLayout( layout41 );

    QWidget* privateLayoutWidget = new QWidget( this, "layout11" );
    privateLayoutWidget->setGeometry( QRect( 20, 660, 616, 34 ) );
    layout11_3 = new QHBoxLayout( privateLayoutWidget, 11, 6, "layout11_3");

    initArmButton = new QPushButton( privateLayoutWidget, "initArmButton" );
    layout11_3->addWidget( initArmButton );

    homeArmButton = new QPushButton( privateLayoutWidget, "homeArmButton" );
    layout11_3->addWidget( homeArmButton );
    spacer7 = new QSpacerItem( 190, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout11_3->addItem( spacer7 );

    quitButton = new QPushButton( privateLayoutWidget, "quitButton" );
    layout11_3->addWidget( quitButton );

    QWidget* privateLayoutWidget_2 = new QWidget( this, "layout12" );
    privateLayoutWidget_2->setGeometry( QRect( 20, 640, 616, 18 ) );
    layout12_2 = new QHBoxLayout( privateLayoutWidget_2, 11, 6, "layout12_2");

    textLabel2 = new QLabel( privateLayoutWidget_2, "textLabel2" );
    textLabel2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabel2->sizePolicy().hasHeightForWidth() ) );
    layout12_2->addWidget( textLabel2 );

    armStatusMsg = new QLabel( privateLayoutWidget_2, "armStatusMsg" );
    layout12_2->addWidget( armStatusMsg );

    groupBox4 = new QGroupBox( this, "groupBox4" );
    groupBox4->setGeometry( QRect( 280, 330, 500, 191 ) );

    QWidget* privateLayoutWidget_3 = new QWidget( groupBox4, "layout32" );
    privateLayoutWidget_3->setGeometry( QRect( 10, 70, 410, 25 ) );
    layout32 = new QHBoxLayout( privateLayoutWidget_3, 0, 0, "layout32");

    parErr = new QLineEdit( privateLayoutWidget_3, "parErr" );
    layout32->addWidget( parErr );

    parCuPos = new QLineEdit( privateLayoutWidget_3, "parCuPos" );
    layout32->addWidget( parCuPos );

    parErrLimit = new QLineEdit( privateLayoutWidget_3, "parErrLimit" );
    layout32->addWidget( parErrLimit );

    parNewPos = new QLineEdit( privateLayoutWidget_3, "parNewPos" );
    layout32->addWidget( parNewPos );

    parSpeed = new QLineEdit( privateLayoutWidget_3, "parSpeed" );
    layout32->addWidget( parSpeed );

    parKP = new QLineEdit( privateLayoutWidget_3, "parKP" );
    layout32->addWidget( parKP );

    parKI = new QLineEdit( privateLayoutWidget_3, "parKI" );
    layout32->addWidget( parKI );

    parKD = new QLineEdit( privateLayoutWidget_3, "parKD" );
    layout32->addWidget( parKD );

    parDeadBand = new QLineEdit( privateLayoutWidget_3, "parDeadBand" );
    layout32->addWidget( parDeadBand );

    parOffset = new QLineEdit( privateLayoutWidget_3, "parOffset" );
    layout32->addWidget( parOffset );

    parMaxForce = new QLineEdit( privateLayoutWidget_3, "parMaxForce" );
    layout32->addWidget( parMaxForce );

    parCurrForce = new QLineEdit( privateLayoutWidget_3, "parCurrForce" );
    layout32->addWidget( parCurrForce );

    parAccTime = new QLineEdit( privateLayoutWidget_3, "parAccTime" );
    layout32->addWidget( parAccTime );

    parUserIO = new QLineEdit( privateLayoutWidget_3, "parUserIO" );
    layout32->addWidget( parUserIO );

    QWidget* privateLayoutWidget_4 = new QWidget( groupBox4, "layout31" );
    privateLayoutWidget_4->setGeometry( QRect( 10, 30, 410, 35 ) );
    layout31 = new QHBoxLayout( privateLayoutWidget_4, 11, 6, "layout31");

    textLabel3 = new QLabel( privateLayoutWidget_4, "textLabel3" );
    layout31->addWidget( textLabel3 );

    textLabel3_2 = new QLabel( privateLayoutWidget_4, "textLabel3_2" );
    layout31->addWidget( textLabel3_2 );

    textLabel3_3 = new QLabel( privateLayoutWidget_4, "textLabel3_3" );
    layout31->addWidget( textLabel3_3 );

    textLabel3_4 = new QLabel( privateLayoutWidget_4, "textLabel3_4" );
    layout31->addWidget( textLabel3_4 );

    textLabel3_5 = new QLabel( privateLayoutWidget_4, "textLabel3_5" );
    layout31->addWidget( textLabel3_5 );

    textLabel3_6 = new QLabel( privateLayoutWidget_4, "textLabel3_6" );
    layout31->addWidget( textLabel3_6 );

    textLabel3_7 = new QLabel( privateLayoutWidget_4, "textLabel3_7" );
    layout31->addWidget( textLabel3_7 );

    textLabel3_8 = new QLabel( privateLayoutWidget_4, "textLabel3_8" );
    layout31->addWidget( textLabel3_8 );

    textLabel3_9 = new QLabel( privateLayoutWidget_4, "textLabel3_9" );
    layout31->addWidget( textLabel3_9 );

    textLabel1_10_3 = new QLabel( privateLayoutWidget_4, "textLabel1_10_3" );
    layout31->addWidget( textLabel1_10_3 );

    textLabel1_10_2 = new QLabel( privateLayoutWidget_4, "textLabel1_10_2" );
    layout31->addWidget( textLabel1_10_2 );

    textLabel1_10 = new QLabel( privateLayoutWidget_4, "textLabel1_10" );
    layout31->addWidget( textLabel1_10 );

    textLabel1_10_4 = new QLabel( privateLayoutWidget_4, "textLabel1_10_4" );
    layout31->addWidget( textLabel1_10_4 );

    textLabel1_10_5 = new QLabel( privateLayoutWidget_4, "textLabel1_10_5" );
    layout31->addWidget( textLabel1_10_5 );
    languageChange();
    resize( QSize(813, 725).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( initArmButton, SIGNAL( clicked() ), this, SLOT( initArm() ) );
    connect( homeArmButton, SIGNAL( clicked() ), this, SLOT( homeArm() ) );
    connect( moveArmButton, SIGNAL( clicked() ), this, SLOT( moveArm() ) );
    connect( zedVal, SIGNAL( valueChanged(int) ), this, SLOT( moveZed(int) ) );
    connect( sholderVal, SIGNAL( valueChanged(int) ), this, SLOT( moveSholder(int) ) );
    connect( elbowVal, SIGNAL( valueChanged(int) ), this, SLOT( moveElbow(int) ) );
    connect( yawVal, SIGNAL( valueChanged(int) ), this, SLOT( moveYaw(int) ) );
    connect( wrist1Val, SIGNAL( valueChanged(int) ), this, SLOT( moveWrist1(int) ) );
    connect( wrist2Val, SIGNAL( valueChanged(int) ), this, SLOT( moveWrist2(int) ) );
    connect( gripperVal, SIGNAL( valueChanged(int) ), this, SLOT( moveGripper(int) ) );
    connect( getJointsPositionButton, SIGNAL( clicked() ), this, SLOT( getCurrentJointPositions() ) );
    connect( quitButton, SIGNAL( clicked() ), this, SLOT( close() ) );
    connect( wristRollVal, SIGNAL( valueChanged(int) ), this, SLOT( wristRoll(int) ) );
    connect( wristTiltVal, SIGNAL( valueChanged(int) ), this, SLOT( wristTilt(int) ) );
    connect( doInterpolationButton, SIGNAL( clicked() ), this, SLOT( doInterpolation() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
RT100ControlForm::~RT100ControlForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void RT100ControlForm::languageChange()
{
    setCaption( tr( "RT100 Arm Control" ) );
    groupBox1->setTitle( tr( "Arm Control" ) );
    textLabel1->setText( tr( "Zed:" ) );
    textLabel1_2->setText( tr( "Sholder:" ) );
    textLabel1_3->setText( tr( "Elbow" ) );
    textLabel1_4->setText( tr( "Yaw:" ) );
    textLabel1_5->setText( tr( "Wrist1:" ) );
    textLabel1_6->setText( tr( "Wrist2:" ) );
    textLabel1_6_2->setText( tr( "Gripper:" ) );
    gripperVal->setSpecialValueText( QString::null );
    textLabel1_7->setText( tr( "Wrist Tilt:" ) );
    textLabel2_2->setText( tr( "Wrist Roll:" ) );
    moveMode->setText( tr( "Move Immediatly" ) );
    moveArmButton->setText( tr( "Move Arm" ) );
    getJointsPositionButton->setText( tr( "Get Joint Positions" ) );
    groupBox2->setTitle( tr( "Interpolation Move" ) );
    textLabel1_8->setText( tr( "Zed:" ) );
    textLabel1_2_3->setText( tr( "Sholder:" ) );
    textLabel1_3_3->setText( tr( "Elbow" ) );
    textLabel1_4_3->setText( tr( "Yaw:" ) );
    textLabel1_5_3->setText( tr( "Wrist1:" ) );
    textLabel1_6_4->setText( tr( "Wrist2:" ) );
    textLabel1_6_2_3->setText( tr( "Gripper:" ) );
    textLabel1_7_3->setText( tr( "Wrist Tilt:" ) );
    textLabel2_2_3->setText( tr( "Wrist Roll:" ) );
    doInterpolationButton->setText( tr( "Interpolation Move" ) );
    groupBox7->setTitle( tr( "Manual Move" ) );
    textLabel2_3->setText( tr( "Stop\n"
"(motors\n"
"powered)" ) );
    textLabel2_3_2->setText( tr( "Forward" ) );
    textLabel2_3_3->setText( tr( "Backward" ) );
    textLabel2_3_4->setText( tr( "Stop\n"
"(motors \n"
"unpowered)" ) );
    textLabel1_9->setText( tr( "Zed:" ) );
    radioButton40->setText( QString::null );
    radioButton40_2->setText( QString::null );
    radioButton40_3->setText( QString::null );
    radioButton40_4->setText( QString::null );
    textLabel1_9_2->setText( tr( "Sholder:" ) );
    radioButton40_5->setText( QString::null );
    radioButton40_2_2->setText( QString::null );
    radioButton40_3_2->setText( QString::null );
    radioButton40_4_2->setText( QString::null );
    textLabel1_9_3->setText( tr( "Elbow:" ) );
    radioButton40_6->setText( QString::null );
    radioButton40_2_3->setText( QString::null );
    radioButton40_3_3->setText( QString::null );
    radioButton40_4_3->setText( QString::null );
    textLabel1_9_4->setText( tr( "Yaw:" ) );
    radioButton40_7->setText( QString::null );
    radioButton40_2_4->setText( QString::null );
    radioButton40_3_4->setText( QString::null );
    radioButton40_4_4->setText( QString::null );
    textLabel1_9_5->setText( tr( "Wrist1:" ) );
    radioButton40_8->setText( QString::null );
    radioButton40_2_5->setText( QString::null );
    radioButton40_3_5->setText( QString::null );
    radioButton40_4_5->setText( QString::null );
    textLabel1_9_6->setText( tr( "Wrist2:" ) );
    radioButton40_9->setText( QString::null );
    radioButton40_2_6->setText( QString::null );
    radioButton40_3_6->setText( QString::null );
    radioButton40_4_6->setText( QString::null );
    textLabel1_9_7->setText( tr( "Gripper:" ) );
    radioButton40_10->setText( QString::null );
    radioButton40_2_7->setText( QString::null );
    radioButton40_3_7->setText( QString::null );
    radioButton40_4_7->setText( QString::null );
    manualMoveBtn->setText( tr( "Move" ) );
    allStopBtn->setText( tr( "All Stop" ) );
    initArmButton->setText( tr( "Initalize Arm" ) );
    homeArmButton->setText( tr( "Home arm" ) );
    quitButton->setText( tr( "Quit" ) );
    textLabel2->setText( tr( "Arm Status:" ) );
    armStatusMsg->setText( tr( "Not initalized" ) );
    groupBox4->setTitle( tr( "Parameters" ) );
    parErr->setInputMask( QString::null );
    textLabel3->setText( tr( "Error" ) );
    textLabel3_2->setText( tr( "CU \n"
"Pos" ) );
    textLabel3_3->setText( tr( "Err\n"
"Limit" ) );
    textLabel3_4->setText( tr( "New\n"
"Pos" ) );
    textLabel3_5->setText( tr( "Speed" ) );
    textLabel3_6->setText( tr( "KP" ) );
    textLabel3_7->setText( tr( "KI" ) );
    textLabel3_8->setText( tr( "KD" ) );
    textLabel3_9->setText( tr( "Dead\n"
"Band" ) );
    textLabel1_10_3->setText( tr( "Offset" ) );
    textLabel1_10_2->setText( tr( "Max\n"
"Force" ) );
    textLabel1_10->setText( tr( "Curr\n"
"Force" ) );
    textLabel1_10_4->setText( tr( "Acc\n"
"Time" ) );
    textLabel1_10_5->setText( tr( "User\n"
"IO" ) );
}

