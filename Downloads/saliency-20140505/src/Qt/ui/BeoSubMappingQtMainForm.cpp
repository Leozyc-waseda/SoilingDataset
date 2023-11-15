/****************************************************************************
** Form implementation generated from reading ui file 'Qt/BeoSubMappingQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/BeoSubMappingQtMainForm.h"

#include <qvariant.h>
#include <qimage.h>
#include <qpixmap.h>
#include <pthread.h>
#include <dirent.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <qpainter.h>
#include <qlabel.h>
#include <qpushbutton.h>
#include <qlistbox.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Raster/Raster.H"
#include "QtUtil/ImageConvert.H"
#include "Qt/poolimage.h"
#include "Qt/BeoSubMappingQtMainForm.ui.h"

/*
 *  Constructs a BeoSubMappingQtMainForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
BeoSubMappingQtMainForm::BeoSubMappingQtMainForm( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "BeoSubMappingQtMainForm" );

    LabelLog = new QLabel( this, "LabelLog" );
    LabelLog->setGeometry( QRect( 140, 10, 33, 20 ) );

    Log = new QLabel( this, "Log" );
    Log->setGeometry( QRect( 20, 40, 300, 120 ) );

    save = new QPushButton( this, "save" );
    save->setGeometry( QRect( 670, 530, 90, 80 ) );

    List = new QListBox( this, "List" );
    List->setGeometry( QRect( 650, 200, 130, 300 ) );
    List->setPaletteForegroundColor( QColor( 85, 0, 255 ) );

    LabelImage = new QLabel( this, "LabelImage" );
    LabelImage->setGeometry( QRect( 390, 10, 48, 20 ) );

    poolImage = new PoolImage( this, "poolImage" );
    poolImage->setGeometry( QRect( 20, 170, 600, 440 ) );

    displayCoord = new QLabel( this, "displayCoord" );
    displayCoord->setGeometry( QRect( 20, 620, 590, 21 ) );

    LabelList = new QLabel( this, "LabelList" );
    LabelList->setGeometry( QRect( 670, 151, 90, 40 ) );

    loadAS = new QPushButton( this, "loadAS" );
    loadAS->setGeometry( QRect( 630, 60, 150, 30 ) );

    load = new QPushButton( this, "load" );
    load->setGeometry( QRect( 630, 110, 150, 30 ) );

    restHeading = new QPushButton( this, "restHeading" );
    restHeading->setGeometry( QRect( 10, 0, 100, 30 ) );

    test = new QLabel( this, "test" );
    test->setGeometry( QRect( 490, 470, 120, 90 ) );
    test->setBackgroundMode( QLabel::PaletteBackground );
    test->setScaledContents( TRUE );

    displayImage = new QLabel( this, "displayImage" );
    displayImage->setGeometry( QRect( 350, 40, 120, 90 ) );
    displayImage->setBackgroundMode( QLabel::PaletteBackground );
    displayImage->setScaledContents( TRUE );

    changeHeading = new QLineEdit( this, "changeHeading" );
    changeHeading->setGeometry( QRect( 480, 133, 140, 23 ) );

    LabelImage_2 = new QLabel( this, "LabelImage_2" );
    LabelImage_2->setGeometry( QRect( 480, 110, 140, 20 ) );

    refresh = new QPushButton( this, "refresh" );
    refresh->setGeometry( QRect( 630, 10, 150, 30 ) );

    change2bottom = new QPushButton( this, "change2bottom" );
    change2bottom->setGeometry( QRect( 470, 70, 150, 30 ) );

    change2front = new QPushButton( this, "change2front" );
    change2front->setGeometry( QRect( 470, 40, 150, 30 ) );

    change2top = new QPushButton( this, "change2top" );
    change2top->setGeometry( QRect( 470, 10, 150, 30 ) );
    languageChange();
    resize( QSize(800, 644).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( List, SIGNAL( selectionChanged(QListBoxItem*) ), this, SLOT( displayLog() ) );
    connect( refresh, SIGNAL( clicked() ), this, SLOT( refreshImages() ) );
    connect( refresh, SIGNAL( clicked() ), this, SLOT( createIcon() ) );
    connect( load, SIGNAL( clicked() ), this, SLOT( loadList() ) );
    connect( save, SIGNAL( clicked() ), this, SLOT( saveiconList() ) );
    connect( loadAS, SIGNAL( clicked() ), this, SLOT( LoadAngleScale() ) );
    connect( restHeading, SIGNAL( clicked() ), this, SLOT( resetAllHeading() ) );
    connect( changeHeading, SIGNAL( returnPressed() ), this, SLOT( changeTheHeading() ) );
    connect( change2bottom, SIGNAL( clicked() ), this, SLOT( change_bottom() ) );
    connect( change2front, SIGNAL( clicked() ), this, SLOT( change_front() ) );
    connect( change2top, SIGNAL( clicked() ), this, SLOT( change_top() ) );
    init();
}

/*
 *  Destroys the object and frees any allocated resources
 */
BeoSubMappingQtMainForm::~BeoSubMappingQtMainForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void BeoSubMappingQtMainForm::languageChange()
{
    setCaption( tr( "BeoSubMappingQtMainForm" ) );
    LabelLog->setText( tr( "<b>Log:</b>" ) );
    Log->setText( QString::null );
    save->setText( tr( "save" ) );
    LabelImage->setText( tr( "<b>Image:</b>" ) );
    displayCoord->setText( QString::null );
    LabelList->setText( tr( "<b>List of images:</b>" ) );
    loadAS->setText( tr( "Load angle and scale" ) );
    load->setText( tr( "load" ) );
    restHeading->setText( tr( "reset\n"
"heading" ) );
    test->setText( QString::null );
    displayImage->setText( QString::null );
    LabelImage_2->setText( tr( "change heading" ) );
    refresh->setText( tr( "refresh" ) );
    change2bottom->setText( tr( "bottom" ) );
    change2front->setText( tr( "front" ) );
    change2top->setText( tr( "top" ) );
}

