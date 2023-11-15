/****************************************************************************
** Form implementation generated from reading ui file 'Qt/BeoSubQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/BeoSubQtMainForm.h"

#include <qvariant.h>
#include <qimage.h>
#include <qpixmap.h>
#include <pthread.h>
#include <dirent.h>
#include <qlabel.h>
#include <qpushbutton.h>
#include <qscrollbar.h>
#include <qradiobutton.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "QtUtil/ImageConvert.H"
#include "Raster/Raster.H"
#include "Qt/BeoSubQtMainForm.ui.h"

/*
 *  Constructs a BeoSubQtMainForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
BeoSubQtMainForm::BeoSubQtMainForm( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "BeoSubQt" );
    setSizeGripEnabled( FALSE );

    bottomfilename = new QLabel( this, "bottomfilename" );
    bottomfilename->setGeometry( QRect( 740, 10, 190, 20 ) );

    saveDown = new QPushButton( this, "saveDown" );
    saveDown->setGeometry( QRect( 730, 30, 205, 20 ) );

    frontfilename = new QLabel( this, "frontfilename" );
    frontfilename->setGeometry( QRect( 410, 10, 190, 20 ) );

    saveFront = new QPushButton( this, "saveFront" );
    saveFront->setGeometry( QRect( 400, 30, 206, 20 ) );

    saveUp = new QPushButton( this, "saveUp" );
    saveUp->setEnabled( TRUE );
    saveUp->setGeometry( QRect( 70, 30, 205, 20 ) );

    topfilename = new QLabel( this, "topfilename" );
    topfilename->setGeometry( QRect( 80, 10, 190, 20 ) );

    textLabel2_2_2_2 = new QLabel( this, "textLabel2_2_2_2" );
    textLabel2_2_2_2->setGeometry( QRect( 360, 600, 30, 30 ) );

    TsDisplay = new QLabel( this, "TsDisplay" );
    TsDisplay->setGeometry( QRect( 310, 600, 40, 30 ) );
    TsDisplay->setFocusPolicy( QLabel::NoFocus );

    DiveScroll = new QScrollBar( this, "DiveScroll" );
    DiveScroll->setGeometry( QRect( 480, 610, 170, 21 ) );
    DiveScroll->setMinValue( 0 );
    DiveScroll->setMaxValue( 100 );
    DiveScroll->setOrientation( QScrollBar::Horizontal );

    FrontBScroll = new QScrollBar( this, "FrontBScroll" );
    FrontBScroll->setGeometry( QRect( 131, 530, 170, 21 ) );
    FrontBScroll->setMinValue( 0 );
    FrontBScroll->setMaxValue( 100 );
    FrontBScroll->setOrientation( QScrollBar::Horizontal );

    textLabel2_2_2 = new QLabel( this, "textLabel2_2_2" );
    textLabel2_2_2->setGeometry( QRect( 10, 640, 80, 30 ) );

    FrontBDisplay = new QLabel( this, "FrontBDisplay" );
    FrontBDisplay->setGeometry( QRect( 310, 520, 40, 30 ) );

    LineEdit12 = new QLabel( this, "LineEdit12" );
    LineEdit12->setGeometry( QRect( 440, 520, 40, 30 ) );

    TsScroll = new QScrollBar( this, "TsScroll" );
    TsScroll->setGeometry( QRect( 130, 610, 170, 21 ) );
    TsScroll->setMinValue( -100 );
    TsScroll->setMaxValue( 100 );
    TsScroll->setOrientation( QScrollBar::Horizontal );

    BsDisplay = new QLabel( this, "BsDisplay" );
    BsDisplay->setGeometry( QRect( 660, 600, 40, 30 ) );

    textLabel2_2_2_3 = new QLabel( this, "textLabel2_2_2_3" );
    textLabel2_2_2_3->setGeometry( QRect( 360, 640, 40, 30 ) );

    RearBScroll = new QScrollBar( this, "RearBScroll" );
    RearBScroll->setGeometry( QRect( 480, 530, 170, 21 ) );
    RearBScroll->setMinValue( 0 );
    RearBScroll->setMaxValue( 100 );
    RearBScroll->setOrientation( QScrollBar::Horizontal );

    OrientScroll = new QScrollBar( this, "OrientScroll" );
    OrientScroll->setGeometry( QRect( 130, 650, 170, 21 ) );
    OrientScroll->setMinValue( -180 );
    OrientScroll->setMaxValue( 180 );
    OrientScroll->setOrientation( QScrollBar::Horizontal );

    LTDisplay = new QLabel( this, "LTDisplay" );
    LTDisplay->setGeometry( QRect( 310, 560, 40, 30 ) );

    textLabel2_2 = new QLabel( this, "textLabel2_2" );
    textLabel2_2->setGeometry( QRect( 10, 600, 80, 30 ) );

    LineEdit16 = new QLabel( this, "LineEdit16" );
    LineEdit16->setGeometry( QRect( 440, 560, 40, 30 ) );

    RTScroll = new QScrollBar( this, "RTScroll" );
    RTScroll->setGeometry( QRect( 480, 570, 170, 21 ) );
    RTScroll->setMinValue( -100 );
    RTScroll->setMaxValue( 100 );
    RTScroll->setOrientation( QScrollBar::Horizontal );

    LTScroll = new QScrollBar( this, "LTScroll" );
    LTScroll->setGeometry( QRect( 130, 570, 170, 21 ) );
    LTScroll->setMinValue( -100 );
    LTScroll->setMaxValue( 100 );
    LTScroll->setOrientation( QScrollBar::Horizontal );

    TiltDisplay = new QLabel( this, "TiltDisplay" );
    TiltDisplay->setGeometry( QRect( 660, 640, 40, 30 ) );

    RTDisplay = new QLabel( this, "RTDisplay" );
    RTDisplay->setGeometry( QRect( 660, 560, 40, 30 ) );

    RearBDisplay = new QLabel( this, "RearBDisplay" );
    RearBDisplay->setGeometry( QRect( 660, 520, 40, 30 ) );

    TurnDisplay = new QLabel( this, "TurnDisplay" );
    TurnDisplay->setGeometry( QRect( 310, 640, 40, 30 ) );

    Label_2 = new QLabel( this, "Label_2" );
    Label_2->setGeometry( QRect( 10, 490, 230, 21 ) );

    textLabel3 = new QLabel( this, "textLabel3" );
    textLabel3->setGeometry( QRect( 10, 410, 70, 30 ) );

    textLabel2 = new QLabel( this, "textLabel2" );
    textLabel2->setGeometry( QRect( 10, 370, 70, 30 ) );

    textLabel3_2 = new QLabel( this, "textLabel3_2" );
    textLabel3_2->setGeometry( QRect( 10, 450, 70, 30 ) );

    textLabel1 = new QLabel( this, "textLabel1" );
    textLabel1->setGeometry( QRect( 10, 330, 70, 30 ) );

    ImagePixmapLabel1 = new QLabel( this, "ImagePixmapLabel1" );
    ImagePixmapLabel1->setGeometry( QRect( 10, 50, 320, 240 ) );
    ImagePixmapLabel1->setBackgroundMode( QLabel::NoBackground );
    ImagePixmapLabel1->setScaledContents( TRUE );

    LineEdit2_2 = new QLabel( this, "LineEdit2_2" );
    LineEdit2_2->setGeometry( QRect( 160, 370, 60, 30 ) );

    LineEdit1 = new QLabel( this, "LineEdit1" );
    LineEdit1->setGeometry( QRect( 90, 330, 60, 30 ) );

    LineEdit2 = new QLabel( this, "LineEdit2" );
    LineEdit2->setGeometry( QRect( 90, 370, 60, 30 ) );

    LineEdit3_2 = new QLabel( this, "LineEdit3_2" );
    LineEdit3_2->setGeometry( QRect( 160, 410, 60, 30 ) );

    LineEdit4 = new QLabel( this, "LineEdit4" );
    LineEdit4->setGeometry( QRect( 90, 450, 60, 30 ) );

    LineEdit3 = new QLabel( this, "LineEdit3" );
    LineEdit3->setGeometry( QRect( 90, 410, 60, 30 ) );

    LineEdit1_2 = new QLabel( this, "LineEdit1_2" );
    LineEdit1_2->setGeometry( QRect( 160, 330, 60, 30 ) );

    LineEdit4_2 = new QLabel( this, "LineEdit4_2" );
    LineEdit4_2->setGeometry( QRect( 160, 450, 60, 30 ) );

    Label_1_2 = new QLabel( this, "Label_1_2" );
    Label_1_2->setGeometry( QRect( 10, 290, 83, 25 ) );

    Label_1_2_2_2 = new QLabel( this, "Label_1_2_2_2" );
    Label_1_2_2_2->setGeometry( QRect( 90, 290, 50, 25 ) );

    Label_1_2_2 = new QLabel( this, "Label_1_2_2" );
    Label_1_2_2->setGeometry( QRect( 160, 290, 50, 25 ) );

    LineEdit11 = new QLabel( this, "LineEdit11" );
    LineEdit11->setGeometry( QRect( 90, 520, 40, 30 ) );

    textLabel11 = new QLabel( this, "textLabel11" );
    textLabel11->setGeometry( QRect( 10, 520, 80, 30 ) );

    LineEdit15 = new QLabel( this, "LineEdit15" );
    LineEdit15->setGeometry( QRect( 90, 560, 40, 30 ) );

    textLabel15 = new QLabel( this, "textLabel15" );
    textLabel15->setGeometry( QRect( 10, 560, 80, 30 ) );

    textLabel16 = new QLabel( this, "textLabel16" );
    textLabel16->setGeometry( QRect( 360, 560, 80, 30 ) );

    textLabel12 = new QLabel( this, "textLabel12" );
    textLabel12->setGeometry( QRect( 360, 520, 80, 30 ) );

    PitchScroll = new QScrollBar( this, "PitchScroll" );
    PitchScroll->setGeometry( QRect( 480, 650, 170, 21 ) );
    PitchScroll->setMinValue( -60 );
    PitchScroll->setMaxValue( 60 );
    PitchScroll->setOrientation( QScrollBar::Horizontal );

    PitchPID = new QRadioButton( this, "PitchPID" );
    PitchPID->setGeometry( QRect( 250, 380, 20, 20 ) );

    HeadingPID = new QRadioButton( this, "HeadingPID" );
    HeadingPID->setGeometry( QRect( 250, 350, 20, 20 ) );

    KILL = new QRadioButton( this, "KILL" );
    KILL->setGeometry( QRect( 250, 440, 20, 20 ) );

    DepthPID = new QRadioButton( this, "DepthPID" );
    DepthPID->setGeometry( QRect( 250, 410, 20, 20 ) );

    LineEdit11_2_2 = new QLabel( this, "LineEdit11_2_2" );
    LineEdit11_2_2->setGeometry( QRect( 270, 380, 70, 20 ) );

    ButtonStop = new QPushButton( this, "ButtonStop" );
    ButtonStop->setGeometry( QRect( 710, 520, 280, 150 ) );

    ImagePixmapLabel2 = new QLabel( this, "ImagePixmapLabel2" );
    ImagePixmapLabel2->setGeometry( QRect( 340, 50, 320, 240 ) );
    ImagePixmapLabel2->setBackgroundMode( QLabel::NoBackground );
    ImagePixmapLabel2->setScaledContents( TRUE );

    LabelTaskControl_2 = new QLabel( this, "LabelTaskControl_2" );
    LabelTaskControl_2->setGeometry( QRect( 870, 300, 110, 30 ) );

    CPUtemp = new QLabel( this, "CPUtemp" );
    CPUtemp->setGeometry( QRect( 870, 340, 100, 20 ) );

    lineEditAdvance = new QLineEdit( this, "lineEditAdvance" );
    lineEditAdvance->setGeometry( QRect( 770, 330, 60, 20 ) );

    lineEditStrafe = new QLineEdit( this, "lineEditStrafe" );
    lineEditStrafe->setGeometry( QRect( 770, 360, 60, 20 ) );

    resetA = new QPushButton( this, "resetA" );
    resetA->setGeometry( QRect( 680, 390, 160, 30 ) );

    GetDirections = new QPushButton( this, "GetDirections" );
    GetDirections->setGeometry( QRect( 680, 430, 160, 41 ) );

    textLabel12_2_2 = new QLabel( this, "textLabel12_2_2" );
    textLabel12_2_2->setGeometry( QRect( 690, 360, 60, 20 ) );

    ButtonTaskB = new QPushButton( this, "ButtonTaskB" );
    ButtonTaskB->setGeometry( QRect( 560, 450, 90, 30 ) );

    ButtonDecode = new QPushButton( this, "ButtonDecode" );
    ButtonDecode->setGeometry( QRect( 560, 370, 90, 30 ) );

    ButtonTaskC = new QPushButton( this, "ButtonTaskC" );
    ButtonTaskC->setGeometry( QRect( 560, 490, 90, 30 ) );

    LabelTaskControl = new QLabel( this, "LabelTaskControl" );
    LabelTaskControl->setGeometry( QRect( 560, 300, 40, 20 ) );

    ButtonTaskA = new QPushButton( this, "ButtonTaskA" );
    ButtonTaskA->setGeometry( QRect( 560, 410, 90, 30 ) );

    DepthD = new QLineEdit( this, "DepthD" );
    DepthD->setGeometry( QRect( 480, 410, 50, 20 ) );

    DepthP = new QLineEdit( this, "DepthP" );
    DepthP->setGeometry( QRect( 360, 410, 50, 20 ) );

    LineEdit11_2_3 = new QLabel( this, "LineEdit11_2_3" );
    LineEdit11_2_3->setGeometry( QRect( 380, 320, 16, 20 ) );

    PitchD = new QLineEdit( this, "PitchD" );
    PitchD->setGeometry( QRect( 480, 380, 50, 20 ) );

    DepthI = new QLineEdit( this, "DepthI" );
    DepthI->setGeometry( QRect( 420, 410, 50, 20 ) );

    PitchI = new QLineEdit( this, "PitchI" );
    PitchI->setGeometry( QRect( 420, 380, 50, 20 ) );

    PitchP = new QLineEdit( this, "PitchP" );
    PitchP->setGeometry( QRect( 360, 380, 50, 20 ) );

    LineEdit11_2_3_2 = new QLabel( this, "LineEdit11_2_3_2" );
    LineEdit11_2_3_2->setGeometry( QRect( 440, 320, 16, 20 ) );

    HeadingI = new QLineEdit( this, "HeadingI" );
    HeadingI->setGeometry( QRect( 420, 350, 50, 20 ) );

    HeadingP = new QLineEdit( this, "HeadingP" );
    HeadingP->setGeometry( QRect( 360, 350, 50, 20 ) );

    LineEdit11_2_3_3 = new QLabel( this, "LineEdit11_2_3_3" );
    LineEdit11_2_3_3->setGeometry( QRect( 500, 320, 16, 20 ) );

    HeadingD = new QLineEdit( this, "HeadingD" );
    HeadingD->setGeometry( QRect( 480, 350, 50, 20 ) );

    LineEdit11_2 = new QLabel( this, "LineEdit11_2" );
    LineEdit11_2->setGeometry( QRect( 270, 350, 87, 20 ) );

    LineEdit11_2_2_2 = new QLabel( this, "LineEdit11_2_2_2" );
    LineEdit11_2_2_2->setGeometry( QRect( 270, 410, 73, 20 ) );

    LineEdit11_2_2_2_2 = new QLabel( this, "LineEdit11_2_2_2_2" );
    LineEdit11_2_2_2_2->setGeometry( QRect( 270, 440, 99, 20 ) );

    LabelTaskControl_2_2 = new QLabel( this, "LabelTaskControl_2_2" );
    LabelTaskControl_2_2->setGeometry( QRect( 870, 370, 110, 20 ) );

    indicator = new QLabel( this, "indicator" );
    indicator->setGeometry( QRect( 870, 400, 100, 20 ) );

    textLabel12_2 = new QLabel( this, "textLabel12_2" );
    textLabel12_2->setGeometry( QRect( 690, 330, 60, 20 ) );

    textLabel12_2_3 = new QLabel( this, "textLabel12_2_3" );
    textLabel12_2_3->setGeometry( QRect( 690, 300, 60, 20 ) );

    ButtonGate = new QPushButton( this, "ButtonGate" );
    ButtonGate->setGeometry( QRect( 560, 330, 90, 30 ) );

    ButtonPic = new QPushButton( this, "ButtonPic" );
    ButtonPic->setGeometry( QRect( 860, 430, 120, 30 ) );

    to = new QLineEdit( this, "to" );
    to->setGeometry( QRect( 770, 300, 60, 20 ) );

    ImagePixmapLabel3 = new QLabel( this, "ImagePixmapLabel3" );
    ImagePixmapLabel3->setGeometry( QRect( 670, 50, 320, 240 ) );
    ImagePixmapLabel3->setBackgroundMode( QLabel::NoBackground );
    ImagePixmapLabel3->setScaledContents( TRUE );
    languageChange();
    resize( QSize(1001, 673).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( saveUp, SIGNAL( clicked() ), this, SLOT( saveImageUp() ) );
    connect( saveFront, SIGNAL( clicked() ), this, SLOT( saveImageFront() ) );
    connect( saveDown, SIGNAL( clicked() ), this, SLOT( saveImageDown() ) );
    connect( ButtonGate, SIGNAL( clicked() ), this, SLOT( taskGate() ) );
    connect( ButtonDecode, SIGNAL( clicked() ), this, SLOT( taskDecode() ) );
    connect( ButtonTaskA, SIGNAL( clicked() ), this, SLOT( taskA() ) );
    connect( ButtonTaskB, SIGNAL( clicked() ), this, SLOT( taskB() ) );
    connect( ButtonTaskC, SIGNAL( clicked() ), this, SLOT( taskC() ) );
    connect( ButtonStop, SIGNAL( clicked() ), this, SLOT( stopAll() ) );
    connect( LTScroll, SIGNAL( valueChanged(int) ), this, SLOT( LThruster_valueChanged(int) ) );
    connect( RTScroll, SIGNAL( valueChanged(int) ), this, SLOT( RThuster_valueChanged(int) ) );
    connect( TsScroll, SIGNAL( valueChanged(int) ), this, SLOT( Thrusters_valueChanged(int) ) );
    connect( DiveScroll, SIGNAL( valueChanged(int) ), this, SLOT( dive_valueChanged(int) ) );
    connect( OrientScroll, SIGNAL( valueChanged(int) ), this, SLOT( Orient_valueChanged(int) ) );
    connect( PitchScroll, SIGNAL( valueChanged(int) ), this, SLOT( Pitch_valueChanged(int) ) );
    connect( FrontBScroll, SIGNAL( valueChanged(int) ), this, SLOT( FrontBallast_valueChanged(int) ) );
    connect( RearBScroll, SIGNAL( valueChanged(int) ), this, SLOT( RearBallast_valueChanged(int) ) );
    connect( HeadingPID, SIGNAL( toggled(bool) ), this, SLOT( HeadingPID_toggled() ) );
    connect( PitchPID, SIGNAL( toggled(bool) ), this, SLOT( PitchPID_toggled() ) );
    connect( DepthPID, SIGNAL( toggled(bool) ), this, SLOT( DepthPID_toggled() ) );
    connect( KILL, SIGNAL( toggled(bool) ), this, SLOT( KILL_toggled() ) );
    connect( lineEditAdvance, SIGNAL( returnPressed() ), this, SLOT( advance_return() ) );
    connect( lineEditStrafe, SIGNAL( returnPressed() ), this, SLOT( strafe_return() ) );
    connect( resetA, SIGNAL( clicked() ), this, SLOT( resetAtt_return() ) );
    connect( GetDirections, SIGNAL( clicked() ), this, SLOT( matchAndDirect() ) );
    connect( to, SIGNAL( returnPressed() ), this, SLOT( turnopen_return() ) );
    connect( ButtonPic, SIGNAL( clicked() ), this, SLOT( togglePic() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
BeoSubQtMainForm::~BeoSubQtMainForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void BeoSubQtMainForm::languageChange()
{
    setCaption( tr( "BeoSub" ) );
    bottomfilename->setText( QString::null );
    saveDown->setText( tr( "Save bottom image" ) );
    frontfilename->setText( QString::null );
    saveFront->setText( tr( "Save front image" ) );
    saveUp->setText( tr( "Save top image" ) );
    topfilename->setText( QString::null );
    textLabel2_2_2_2->setText( tr( "Dive" ) );
    TsDisplay->setText( QString::null );
    textLabel2_2_2->setText( tr( "Orient" ) );
    FrontBDisplay->setText( QString::null );
    LineEdit12->setText( QString::null );
    BsDisplay->setText( QString::null );
    textLabel2_2_2_3->setText( tr( "Pitch" ) );
    LTDisplay->setText( QString::null );
    textLabel2_2->setText( tr( "Advance" ) );
    LineEdit16->setText( QString::null );
    TiltDisplay->setText( QString::null );
    RTDisplay->setText( QString::null );
    RearBDisplay->setText( QString::null );
    TurnDisplay->setText( QString::null );
    Label_2->setText( tr( "<b>Ballasts and Thrusters<font size=\"+1\"></font></b>" ) );
    textLabel3->setText( tr( "Heading =" ) );
    textLabel2->setText( tr( "Pitch =" ) );
    textLabel3_2->setText( tr( "Depth =" ) );
    textLabel1->setText( tr( "Roll =" ) );
    LineEdit2_2->setText( QString::null );
    LineEdit1->setText( QString::null );
    LineEdit2->setText( QString::null );
    LineEdit3_2->setText( QString::null );
    LineEdit4->setText( QString::null );
    LineEdit3->setText( QString::null );
    LineEdit1_2->setText( QString::null );
    LineEdit4_2->setText( QString::null );
    Label_1_2->setText( tr( "<b>Compass<font size=\"+1\"></font></b>" ) );
    Label_1_2_2_2->setText( tr( "current" ) );
    Label_1_2_2->setText( tr( "target" ) );
    LineEdit11->setText( QString::null );
    textLabel11->setText( tr( "Front Ballast =" ) );
    LineEdit15->setText( QString::null );
    textLabel15->setText( tr( "L Thruster" ) );
    textLabel16->setText( tr( "R Thruster" ) );
    textLabel12->setText( tr( "Rear Ballast =" ) );
    PitchPID->setText( tr( "radioButton1" ) );
    HeadingPID->setText( tr( "radioButton1" ) );
    KILL->setText( tr( "radioButton1" ) );
    DepthPID->setText( tr( "radioButton1" ) );
    LineEdit11_2_2->setText( tr( "Pitch PID" ) );
    ButtonStop->setText( tr( "Stop\n"
"Everything" ) );
    LabelTaskControl_2->setText( tr( "CPU temperature" ) );
    CPUtemp->setText( QString::null );
    resetA->setText( tr( "reset Attitude" ) );
    GetDirections->setText( tr( "Get Directions from \n"
"Current to Goal" ) );
    textLabel12_2_2->setText( tr( "Strafe" ) );
    ButtonTaskB->setText( tr( "Task B" ) );
    ButtonDecode->setText( tr( "Decode" ) );
    ButtonTaskC->setText( tr( "Task C" ) );
    LabelTaskControl->setText( tr( "Task" ) );
    ButtonTaskA->setText( tr( "Task A" ) );
    LineEdit11_2_3->setText( tr( "P" ) );
    LineEdit11_2_3_2->setText( tr( "I" ) );
    LineEdit11_2_3_3->setText( tr( "D" ) );
    LineEdit11_2->setText( tr( "Heading PID" ) );
    LineEdit11_2_2_2->setText( tr( "Depth PID" ) );
    LineEdit11_2_2_2_2->setText( tr( "Use KillSwitch" ) );
    LabelTaskControl_2_2->setText( tr( "indicator" ) );
    indicator->setText( QString::null );
    textLabel12_2->setText( tr( "Advance" ) );
    textLabel12_2_3->setText( tr( "turnopen" ) );
    ButtonGate->setText( tr( "Gate" ) );
    ButtonPic->setText( tr( "Pic?" ) );
}

