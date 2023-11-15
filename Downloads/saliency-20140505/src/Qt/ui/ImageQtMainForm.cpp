/****************************************************************************
** Form implementation generated from reading ui file 'Qt/ImageQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/ImageQtMainForm.h"

#include <qvariant.h>
#include <qfiledialog.h>
#include <qlineedit.h>
#include <qimage.h>
#include <qpixmap.h>
#include <qmessagebox.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcheckbox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qimage.h>
#include <qpixmap.h>

#include "Raster/Raster.H"
#include "QtUtil/ImageConvert.H"
#include "Qt/ImageQtMainForm.ui.h"
/*
 *  Constructs a ImageQtMainForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ImageQtMainForm::ImageQtMainForm( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "ImageQtMainForm" );

    ChooseButton = new QPushButton( this, "ChooseButton" );
    ChooseButton->setGeometry( QRect( 40, 90, 140, 31 ) );

    ImageLoadButton = new QPushButton( this, "ImageLoadButton" );
    ImageLoadButton->setGeometry( QRect( 230, 90, 140, 31 ) );

    ImageFileLineEdit = new QLineEdit( this, "ImageFileLineEdit" );
    ImageFileLineEdit->setGeometry( QRect( 20, 40, 370, 24 ) );

    ImageFileTextLabel = new QLabel( this, "ImageFileTextLabel" );
    ImageFileTextLabel->setGeometry( QRect( 20, 10, 90, 21 ) );

    FullSizeBox = new QCheckBox( this, "FullSizeBox" );
    FullSizeBox->setGeometry( QRect( 190, 140, 210, 31 ) );

    DisplayButton = new QPushButton( this, "DisplayButton" );
    DisplayButton->setGeometry( QRect( 40, 140, 140, 31 ) );

    ImagePixmapLabel = new QLabel( this, "ImagePixmapLabel" );
    ImagePixmapLabel->setGeometry( QRect( 50, 200, 320, 240 ) );
    ImagePixmapLabel->setScaledContents( TRUE );
    languageChange();
    resize( QSize(411, 465).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( ChooseButton, SIGNAL( clicked() ), this, SLOT( setImageFile() ) );
    connect( DisplayButton, SIGNAL( clicked() ), this, SLOT( displayImage() ) );
    connect( ImageLoadButton, SIGNAL( clicked() ), this, SLOT( loadImage() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
ImageQtMainForm::~ImageQtMainForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void ImageQtMainForm::languageChange()
{
    setCaption( tr( "test-ImageQt" ) );
    ChooseButton->setText( tr( "Choose File..." ) );
    ImageLoadButton->setText( tr( "Load" ) );
    ImageFileTextLabel->setText( tr( "Image file:" ) );
    FullSizeBox->setText( tr( "full-size (another window)" ) );
    DisplayButton->setText( tr( "Display" ) );
}

