/****************************************************************************
** Form implementation generated from reading ui file 'Qt/SSCMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/SSCMainForm.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qradiobutton.h>
#include <qgroupbox.h>
#include <qslider.h>
#include <qlcdnumber.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qimage.h>
#include <qpixmap.h>

#include "Qt/SSCMainForm.ui.h"
/*
 *  Constructs a SSCMainForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
SSCMainForm::SSCMainForm( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "SSCMainForm" );

    QWidget* privateLayoutWidget = new QWidget( this, "layoutBottomBar" );
    privateLayoutWidget->setGeometry( QRect( 10, 380, 598, 50 ) );
    layoutBottomBar = new QHBoxLayout( privateLayoutWidget, 11, 6, "layoutBottomBar");

    labelSerDev = new QLabel( privateLayoutWidget, "labelSerDev" );
    labelSerDev->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, labelSerDev->sizePolicy().hasHeightForWidth() ) );
    labelSerDev->setMinimumSize( QSize( 0, 30 ) );
    layoutBottomBar->addWidget( labelSerDev );

    lineEditSerDev = new QLineEdit( privateLayoutWidget, "lineEditSerDev" );
    lineEditSerDev->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, lineEditSerDev->sizePolicy().hasHeightForWidth() ) );
    lineEditSerDev->setMinimumSize( QSize( 0, 30 ) );
    layoutBottomBar->addWidget( lineEditSerDev );
    spacerSerDev = new QSpacerItem( 16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layoutBottomBar->addItem( spacerSerDev );

    labelBaudrate = new QLabel( privateLayoutWidget, "labelBaudrate" );
    labelBaudrate->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, labelBaudrate->sizePolicy().hasHeightForWidth() ) );
    labelBaudrate->setMinimumSize( QSize( 0, 30 ) );
    layoutBottomBar->addWidget( labelBaudrate );

    lineEditBaudrate = new QLineEdit( privateLayoutWidget, "lineEditBaudrate" );
    lineEditBaudrate->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, lineEditBaudrate->sizePolicy().hasHeightForWidth() ) );
    lineEditBaudrate->setMinimumSize( QSize( 50, 30 ) );
    lineEditBaudrate->setMaximumSize( QSize( 50, 32767 ) );
    layoutBottomBar->addWidget( lineEditBaudrate );
    spacerBaud = new QSpacerItem( 16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layoutBottomBar->addItem( spacerBaud );

    radioButtonDec = new QRadioButton( privateLayoutWidget, "radioButtonDec" );
    radioButtonDec->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, radioButtonDec->sizePolicy().hasHeightForWidth() ) );
    radioButtonDec->setMinimumSize( QSize( 0, 30 ) );
    radioButtonDec->setChecked( TRUE );
    layoutBottomBar->addWidget( radioButtonDec );

    radioButtonHex = new QRadioButton( privateLayoutWidget, "radioButtonHex" );
    radioButtonHex->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, radioButtonHex->sizePolicy().hasHeightForWidth() ) );
    radioButtonHex->setMinimumSize( QSize( 0, 30 ) );
    layoutBottomBar->addWidget( radioButtonHex );
    spacerHex = new QSpacerItem( 16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layoutBottomBar->addItem( spacerHex );

    pushButtonQuit = new QPushButton( privateLayoutWidget, "pushButtonQuit" );
    pushButtonQuit->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, pushButtonQuit->sizePolicy().hasHeightForWidth() ) );
    pushButtonQuit->setMinimumSize( QSize( 0, 30 ) );
    pushButtonQuit->setCursor( QCursor( 13 ) );
    pushButtonQuit->setFocusPolicy( QPushButton::ClickFocus );
    pushButtonQuit->setAutoDefault( FALSE );
    pushButtonQuit->setDefault( FALSE );
    layoutBottomBar->addWidget( pushButtonQuit );

    groupBoxAllSSC = new QGroupBox( this, "groupBoxAllSSC" );
    groupBoxAllSSC->setGeometry( QRect( 10, 10, 600, 370 ) );
    groupBoxAllSSC->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)7, 0, 0, groupBoxAllSSC->sizePolicy().hasHeightForWidth() ) );

    QWidget* privateLayoutWidget_2 = new QWidget( groupBoxAllSSC, "layoutAllSSC" );
    privateLayoutWidget_2->setGeometry( QRect( 10, 20, 580, 341 ) );
    layoutAllSSC = new QVBoxLayout( privateLayoutWidget_2, 11, 6, "layoutAllSSC");

    layoutSSC1 = new QHBoxLayout( 0, 0, 6, "layoutSSC1");

    textLabelSSC1 = new QLabel( privateLayoutWidget_2, "textLabelSSC1" );
    textLabelSSC1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC1->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC1_font(  textLabelSSC1->font() );
    textLabelSSC1_font.setBold( TRUE );
    textLabelSSC1->setFont( textLabelSSC1_font );
    layoutSSC1->addWidget( textLabelSSC1 );

    sliderSSC1 = new QSlider( privateLayoutWidget_2, "sliderSSC1" );
    sliderSSC1->setMaxValue( 255 );
    sliderSSC1->setOrientation( QSlider::Horizontal );
    sliderSSC1->setTickmarks( QSlider::Right );
    sliderSSC1->setTickInterval( 5 );
    layoutSSC1->addWidget( sliderSSC1 );

    lCDNumberSSC1 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC1" );
    lCDNumberSSC1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC1->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC1->setMargin( 1 );
    lCDNumberSSC1->setNumDigits( 3 );
    lCDNumberSSC1->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC1->addWidget( lCDNumberSSC1 );
    layoutAllSSC->addLayout( layoutSSC1 );
    spacer1_2 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    layoutAllSSC->addItem( spacer1_2 );

    layoutSSC2 = new QHBoxLayout( 0, 0, 6, "layoutSSC2");

    textLabelSSC2 = new QLabel( privateLayoutWidget_2, "textLabelSSC2" );
    textLabelSSC2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC2->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC2_font(  textLabelSSC2->font() );
    textLabelSSC2_font.setBold( TRUE );
    textLabelSSC2->setFont( textLabelSSC2_font );
    layoutSSC2->addWidget( textLabelSSC2 );

    sliderSSC2 = new QSlider( privateLayoutWidget_2, "sliderSSC2" );
    sliderSSC2->setMaxValue( 255 );
    sliderSSC2->setOrientation( QSlider::Horizontal );
    sliderSSC2->setTickmarks( QSlider::Right );
    sliderSSC2->setTickInterval( 5 );
    layoutSSC2->addWidget( sliderSSC2 );

    lCDNumberSSC2 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC2" );
    lCDNumberSSC2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC2->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC2->setMargin( 1 );
    lCDNumberSSC2->setNumDigits( 3 );
    lCDNumberSSC2->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC2->addWidget( lCDNumberSSC2 );
    layoutAllSSC->addLayout( layoutSSC2 );
    spacer2_3 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    layoutAllSSC->addItem( spacer2_3 );

    layoutSSC3 = new QHBoxLayout( 0, 0, 6, "layoutSSC3");

    textLabelSSC3 = new QLabel( privateLayoutWidget_2, "textLabelSSC3" );
    textLabelSSC3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC3->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC3_font(  textLabelSSC3->font() );
    textLabelSSC3_font.setBold( TRUE );
    textLabelSSC3->setFont( textLabelSSC3_font );
    layoutSSC3->addWidget( textLabelSSC3 );

    sliderSSC3 = new QSlider( privateLayoutWidget_2, "sliderSSC3" );
    sliderSSC3->setMaxValue( 255 );
    sliderSSC3->setOrientation( QSlider::Horizontal );
    sliderSSC3->setTickmarks( QSlider::Right );
    sliderSSC3->setTickInterval( 5 );
    layoutSSC3->addWidget( sliderSSC3 );

    lCDNumberSSC3 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC3" );
    lCDNumberSSC3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC3->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC3->setMargin( 1 );
    lCDNumberSSC3->setNumDigits( 3 );
    lCDNumberSSC3->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC3->addWidget( lCDNumberSSC3 );
    layoutAllSSC->addLayout( layoutSSC3 );
    spacer3_4 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    layoutAllSSC->addItem( spacer3_4 );

    layoutSSC4 = new QHBoxLayout( 0, 0, 6, "layoutSSC4");

    textLabelSSC4 = new QLabel( privateLayoutWidget_2, "textLabelSSC4" );
    textLabelSSC4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC4->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC4_font(  textLabelSSC4->font() );
    textLabelSSC4_font.setBold( TRUE );
    textLabelSSC4->setFont( textLabelSSC4_font );
    layoutSSC4->addWidget( textLabelSSC4 );

    sliderSSC4 = new QSlider( privateLayoutWidget_2, "sliderSSC4" );
    sliderSSC4->setMaxValue( 255 );
    sliderSSC4->setOrientation( QSlider::Horizontal );
    sliderSSC4->setTickmarks( QSlider::Right );
    sliderSSC4->setTickInterval( 5 );
    layoutSSC4->addWidget( sliderSSC4 );

    lCDNumberSSC4 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC4" );
    lCDNumberSSC4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC4->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC4->setMargin( 1 );
    lCDNumberSSC4->setNumDigits( 3 );
    lCDNumberSSC4->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC4->addWidget( lCDNumberSSC4 );
    layoutAllSSC->addLayout( layoutSSC4 );
    spacer4_5 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    layoutAllSSC->addItem( spacer4_5 );

    layoutSSC5 = new QHBoxLayout( 0, 0, 6, "layoutSSC5");

    textLabelSSC5 = new QLabel( privateLayoutWidget_2, "textLabelSSC5" );
    textLabelSSC5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC5->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC5_font(  textLabelSSC5->font() );
    textLabelSSC5_font.setBold( TRUE );
    textLabelSSC5->setFont( textLabelSSC5_font );
    layoutSSC5->addWidget( textLabelSSC5 );

    sliderSSC5 = new QSlider( privateLayoutWidget_2, "sliderSSC5" );
    sliderSSC5->setMaxValue( 255 );
    sliderSSC5->setOrientation( QSlider::Horizontal );
    sliderSSC5->setTickmarks( QSlider::Right );
    sliderSSC5->setTickInterval( 5 );
    layoutSSC5->addWidget( sliderSSC5 );

    lCDNumberSSC5 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC5" );
    lCDNumberSSC5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC5->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC5->setMargin( 1 );
    lCDNumberSSC5->setNumDigits( 3 );
    lCDNumberSSC5->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC5->addWidget( lCDNumberSSC5 );
    layoutAllSSC->addLayout( layoutSSC5 );
    spacer5_6 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    layoutAllSSC->addItem( spacer5_6 );

    layoutSSC6 = new QHBoxLayout( 0, 0, 6, "layoutSSC6");

    textLabelSSC6 = new QLabel( privateLayoutWidget_2, "textLabelSSC6" );
    textLabelSSC6->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC6->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC6_font(  textLabelSSC6->font() );
    textLabelSSC6_font.setBold( TRUE );
    textLabelSSC6->setFont( textLabelSSC6_font );
    layoutSSC6->addWidget( textLabelSSC6 );

    sliderSSC6 = new QSlider( privateLayoutWidget_2, "sliderSSC6" );
    sliderSSC6->setMaxValue( 255 );
    sliderSSC6->setOrientation( QSlider::Horizontal );
    sliderSSC6->setTickmarks( QSlider::Right );
    sliderSSC6->setTickInterval( 5 );
    layoutSSC6->addWidget( sliderSSC6 );

    lCDNumberSSC6 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC6" );
    lCDNumberSSC6->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC6->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC6->setMargin( 1 );
    lCDNumberSSC6->setNumDigits( 3 );
    lCDNumberSSC6->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC6->addWidget( lCDNumberSSC6 );
    layoutAllSSC->addLayout( layoutSSC6 );
    spacer6_7 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    layoutAllSSC->addItem( spacer6_7 );

    layoutSSC7 = new QHBoxLayout( 0, 0, 6, "layoutSSC7");

    textLabelSSC7 = new QLabel( privateLayoutWidget_2, "textLabelSSC7" );
    textLabelSSC7->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC7->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC7_font(  textLabelSSC7->font() );
    textLabelSSC7_font.setBold( TRUE );
    textLabelSSC7->setFont( textLabelSSC7_font );
    layoutSSC7->addWidget( textLabelSSC7 );

    sliderSSC7 = new QSlider( privateLayoutWidget_2, "sliderSSC7" );
    sliderSSC7->setMaxValue( 255 );
    sliderSSC7->setOrientation( QSlider::Horizontal );
    sliderSSC7->setTickmarks( QSlider::Right );
    sliderSSC7->setTickInterval( 5 );
    layoutSSC7->addWidget( sliderSSC7 );

    lCDNumberSSC7 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC7" );
    lCDNumberSSC7->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC7->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC7->setMargin( 1 );
    lCDNumberSSC7->setNumDigits( 3 );
    lCDNumberSSC7->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC7->addWidget( lCDNumberSSC7 );
    layoutAllSSC->addLayout( layoutSSC7 );
    spacer7_8 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    layoutAllSSC->addItem( spacer7_8 );

    layoutSSC8 = new QHBoxLayout( 0, 0, 6, "layoutSSC8");

    textLabelSSC8 = new QLabel( privateLayoutWidget_2, "textLabelSSC8" );
    textLabelSSC8->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabelSSC8->sizePolicy().hasHeightForWidth() ) );
    QFont textLabelSSC8_font(  textLabelSSC8->font() );
    textLabelSSC8_font.setBold( TRUE );
    textLabelSSC8->setFont( textLabelSSC8_font );
    layoutSSC8->addWidget( textLabelSSC8 );

    sliderSSC8 = new QSlider( privateLayoutWidget_2, "sliderSSC8" );
    sliderSSC8->setMaxValue( 255 );
    sliderSSC8->setOrientation( QSlider::Horizontal );
    sliderSSC8->setTickmarks( QSlider::Right );
    sliderSSC8->setTickInterval( 5 );
    layoutSSC8->addWidget( sliderSSC8 );

    lCDNumberSSC8 = new QLCDNumber( privateLayoutWidget_2, "lCDNumberSSC8" );
    lCDNumberSSC8->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)1, 0, 0, lCDNumberSSC8->sizePolicy().hasHeightForWidth() ) );
    lCDNumberSSC8->setMargin( 1 );
    lCDNumberSSC8->setNumDigits( 3 );
    lCDNumberSSC8->setSegmentStyle( QLCDNumber::Flat );
    layoutSSC8->addWidget( lCDNumberSSC8 );
    layoutAllSSC->addLayout( layoutSSC8 );
    languageChange();
    resize( QSize(617, 440).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( sliderSSC1, SIGNAL( valueChanged(int) ), lCDNumberSSC1, SLOT( display(int) ) );
    connect( sliderSSC2, SIGNAL( valueChanged(int) ), lCDNumberSSC2, SLOT( display(int) ) );
    connect( sliderSSC3, SIGNAL( valueChanged(int) ), lCDNumberSSC3, SLOT( display(int) ) );
    connect( sliderSSC4, SIGNAL( valueChanged(int) ), lCDNumberSSC4, SLOT( display(int) ) );
    connect( sliderSSC5, SIGNAL( valueChanged(int) ), lCDNumberSSC5, SLOT( display(int) ) );
    connect( sliderSSC6, SIGNAL( valueChanged(int) ), lCDNumberSSC6, SLOT( display(int) ) );
    connect( sliderSSC7, SIGNAL( valueChanged(int) ), lCDNumberSSC7, SLOT( display(int) ) );
    connect( sliderSSC8, SIGNAL( valueChanged(int) ), lCDNumberSSC8, SLOT( display(int) ) );
    connect( pushButtonQuit, SIGNAL( clicked() ), this, SLOT( close() ) );
    connect( lineEditBaudrate, SIGNAL( textChanged(const QString&) ), this, SLOT( lineEditBaudrate_textChanged(const QString&) ) );
    connect( sliderSSC1, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC1_valueChanged(int) ) );
    connect( sliderSSC2, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC2_valueChanged(int) ) );
    connect( sliderSSC3, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC3_valueChanged(int) ) );
    connect( sliderSSC4, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC4_valueChanged(int) ) );
    connect( sliderSSC5, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC5_valueChanged(int) ) );
    connect( sliderSSC6, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC6_valueChanged(int) ) );
    connect( sliderSSC7, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC7_valueChanged(int) ) );
    connect( sliderSSC8, SIGNAL( valueChanged(int) ), this, SLOT( sliderSSC8_valueChanged(int) ) );
    connect( radioButtonDec, SIGNAL( clicked() ), this, SLOT( radioButtonDec_clicked() ) );
    connect( radioButtonHex, SIGNAL( clicked() ), this, SLOT( radioButtonHex_clicked() ) );
    connect( lineEditSerDev, SIGNAL( returnPressed() ), this, SLOT( lineEditSerDev_returnPressed() ) );
    connect( lineEditBaudrate, SIGNAL( returnPressed() ), this, SLOT( lineEditBaudrate_returnPressed() ) );
    connect( lineEditSerDev, SIGNAL( textChanged(const QString&) ), this, SLOT( lineEditSerDev_textChanged(const QString&) ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
SSCMainForm::~SSCMainForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void SSCMainForm::languageChange()
{
    setCaption( tr( "Serial Servo Controller (Mini-SSC II) Interface" ) );
    labelSerDev->setText( tr( "Serial Dev" ) );
    lineEditSerDev->setText( tr( "/dev/ttyS0" ) );
    QToolTip::add( lineEditSerDev, tr( "Device used to communicate with SSC" ) );
    labelBaudrate->setText( tr( "BaudRate" ) );
    lineEditBaudrate->setText( tr( "9600" ) );
    QToolTip::add( lineEditBaudrate, tr( "Baudrate used to communicate with SSC, typically 2400 or 9600" ) );
    radioButtonDec->setText( tr( "Dec" ) );
    QToolTip::add( radioButtonDec, tr( "Click for decimal displays" ) );
    radioButtonHex->setText( tr( "Hex" ) );
    QToolTip::add( radioButtonHex, tr( "Click for hexadecimal displays" ) );
    pushButtonQuit->setText( tr( "Quit" ) );
    QToolTip::add( pushButtonQuit, tr( "Click to quit!" ) );
    groupBoxAllSSC->setTitle( tr( "Mini-SSC II Controls" ) );
    textLabelSSC1->setText( tr( "SSC #1" ) );
    textLabelSSC2->setText( tr( "SSC #2" ) );
    textLabelSSC3->setText( tr( "SSC #3" ) );
    textLabelSSC4->setText( tr( "SSC #4" ) );
    textLabelSSC5->setText( tr( "SSC #5" ) );
    textLabelSSC6->setText( tr( "SSC #6" ) );
    textLabelSSC7->setText( tr( "SSC #7" ) );
    textLabelSSC8->setText( tr( "SSC #8" ) );
}

