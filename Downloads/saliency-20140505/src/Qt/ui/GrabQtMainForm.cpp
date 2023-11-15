/****************************************************************************
** Form implementation generated from reading ui file 'Qt/GrabQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/GrabQtMainForm.h"

#include <qvariant.h>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qimage.h>
#include <qpixmap.h>
#include <qlabel.h>
#include <qslider.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "QtUtil/ImageConvert.H"
#include "Raster/Raster.H"
#include "Qt/GrabQtMainForm.ui.h"

/*
 *  Constructs a GrabQtMainForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
GrabQtMainForm::GrabQtMainForm( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "GrabQtMainForm" );

    tiltLabel = new QLabel( this, "tiltLabel" );
    tiltLabel->setGeometry( QRect( 180, 10, 30, 20 ) );
    tiltLabel->setFrameShape( QLabel::NoFrame );
    tiltLabel->setFrameShadow( QLabel::Plain );

    panLabel = new QLabel( this, "panLabel" );
    panLabel->setGeometry( QRect( 90, 160, 30, 20 ) );

    panSlider = new QSlider( this, "panSlider" );
    panSlider->setGeometry( QRect( 50, 130, 100, 24 ) );
    panSlider->setOrientation( QSlider::Horizontal );

    pauseButton = new QPushButton( this, "pauseButton" );
    pauseButton->setGeometry( QRect( 40, 30, 120, 30 ) );

    saveButton = new QPushButton( this, "saveButton" );
    saveButton->setGeometry( QRect( 40, 75, 120, 30 ) );

    tiltSlider = new QSlider( this, "tiltSlider" );
    tiltSlider->setGeometry( QRect( 180, 30, 22, 100 ) );
    tiltSlider->setOrientation( QSlider::Vertical );
    languageChange();
    resize( QSize(236, 207).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( panSlider, SIGNAL( valueChanged(int) ), this, SLOT( handlePanSlider(int) ) );
    connect( tiltSlider, SIGNAL( valueChanged(int) ), this, SLOT( handleTiltSlider(int) ) );
    connect( saveButton, SIGNAL( clicked() ), this, SLOT( saveImage() ) );
    connect( pauseButton, SIGNAL( clicked() ), this, SLOT( handlePauseButton() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
GrabQtMainForm::~GrabQtMainForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void GrabQtMainForm::languageChange()
{
    setCaption( tr( "test-GrabQt" ) );
    tiltLabel->setText( tr( "Tilt" ) );
    panLabel->setText( tr( "Pan" ) );
    pauseButton->setText( tr( "Grab" ) );
    saveButton->setText( tr( "Save..." ) );
}

