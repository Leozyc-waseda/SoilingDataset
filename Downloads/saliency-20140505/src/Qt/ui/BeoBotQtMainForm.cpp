/****************************************************************************
** Form implementation generated from reading ui file 'Qt/BeoBotQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/BeoBotQtMainForm.h"

#include <qvariant.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Raster/Raster.H"
#include "Qt/beobotmap.h"
#include "Qt/BeoBotQtMainForm.ui.h"

/*
 *  Constructs a BeoBotQtMainForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
BeoBotQtMainForm::BeoBotQtMainForm( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
        setName( "BeoBotQtMainForm" );

    MapLabel = new QLabel( this, "MapLabel" );
    MapLabel->setGeometry( QRect( 360, 10, 70, 20 ) );

    CameraLabel = new QLabel( this, "CameraLabel" );
    CameraLabel->setGeometry( QRect( 20, 10, 70, 20 ) );

    DisplayLabel = new QLabel( this, "DisplayLabel" );
    DisplayLabel->setGeometry( QRect( 20, 280, 70, 20 ) );

    ParameterLabel = new QLabel( this, "ParameterLabel" );
    ParameterLabel->setGeometry( QRect( 360, 280, 110, 20 ) );

    map = new BeoBotMap( this, "map" );
    map->setGeometry( QRect( 380, 40, 200, 200 ) );

    displayCoord = new QLabel( this, "displayCoord" );
    displayCoord->setGeometry( QRect( 360, 250, 110, 20 ) );
    displayCoord->setScaledContents( FALSE );

    displayImage3 = new QLabel( this, "displayImage3" );
    displayImage3->setGeometry( QRect( 20, 140, 120, 90 ) );
    displayImage3->setBackgroundMode( QLabel::PaletteBackground );
    displayImage3->setScaledContents( TRUE );

    displayImage4 = new QLabel( this, "displayImage4" );
    displayImage4->setGeometry( QRect( 150, 140, 120, 90 ) );
    displayImage4->setBackgroundMode( QLabel::PaletteBackground );
    displayImage4->setScaledContents( TRUE );

    displayImage2 = new QLabel( this, "displayImage2" );
    displayImage2->setGeometry( QRect( 150, 40, 120, 90 ) );
    displayImage2->setBackgroundMode( QLabel::PaletteBackground );
    displayImage2->setScaledContents( TRUE );

    displayImage1 = new QLabel( this, "displayImage1" );
    displayImage1->setGeometry( QRect( 20, 40, 120, 90 ) );
    displayImage1->setBackgroundMode( QLabel::PaletteBackground );
    displayImage1->setScaledContents( TRUE );
    languageChange();
    resize( QSize(600, 480).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
    init();
}

/*
 *  Destroys the object and frees any allocated resources
 */
BeoBotQtMainForm::~BeoBotQtMainForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void BeoBotQtMainForm::languageChange()
{
    setCaption( tr( "BeoBotQt" ) );
    MapLabel->setText( tr( "<b>Map:</b>" ) );
    CameraLabel->setText( tr( "<b>Cameras:</b>" ) );
    DisplayLabel->setText( tr( "<b>display:</b>" ) );
    ParameterLabel->setText( tr( "<b>Parameters:</b>" ) );
    displayCoord->setText( QString::null );
    displayImage3->setText( QString::null );
    displayImage4->setText( QString::null );
    displayImage2->setText( QString::null );
    displayImage1->setText( QString::null );
}

