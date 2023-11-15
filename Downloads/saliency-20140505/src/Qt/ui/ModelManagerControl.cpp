/****************************************************************************
** Form implementation generated from reading ui file 'Qt/ModelManagerControl.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/ModelManagerControl.h"

#include <qvariant.h>
#include <qfiledialog.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Qt/ModelManagerControl.ui.h"

/*
 *  Constructs a ModelManagerControl as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ModelManagerControl::ModelManagerControl( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "ModelManagerControl" );

    saveButton = new QPushButton( this, "saveButton" );
    saveButton->setGeometry( QRect( 30, 130, 120, 40 ) );
    saveButton->setAutoDefault( FALSE );

    loadButton = new QPushButton( this, "loadButton" );
    loadButton->setGeometry( QRect( 30, 80, 120, 40 ) );
    loadButton->setAutoDefault( FALSE );

    startstopButton = new QPushButton( this, "startstopButton" );
    startstopButton->setGeometry( QRect( 30, 210, 120, 40 ) );
    startstopButton->setAutoDefault( TRUE );
    startstopButton->setDefault( TRUE );

    exitButton = new QPushButton( this, "exitButton" );
    exitButton->setGeometry( QRect( 30, 290, 120, 40 ) );

    configButton = new QPushButton( this, "configButton" );
    configButton->setGeometry( QRect( 30, 20, 120, 40 ) );
    configButton->setAutoDefault( FALSE );
    languageChange();
    resize( QSize(182, 353).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( configButton, SIGNAL( clicked() ), this, SLOT( showConfigDialog() ) );
    connect( loadButton, SIGNAL( clicked() ), this, SLOT( loadConfig() ) );
    connect( saveButton, SIGNAL( clicked() ), this, SLOT( saveConfig() ) );
    connect( startstopButton, SIGNAL( clicked() ), this, SLOT( start_or_stop() ) );
    connect( exitButton, SIGNAL( clicked() ), this, SLOT( exitPressed() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
ModelManagerControl::~ModelManagerControl()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void ModelManagerControl::languageChange()
{
    setCaption( tr( "Control Panel" ) );
    saveButton->setText( tr( "Save config..." ) );
    loadButton->setText( tr( "Load config..." ) );
    startstopButton->setText( tr( "Start" ) );
    exitButton->setText( tr( "Exit" ) );
    configButton->setText( tr( "Configure..." ) );
}

