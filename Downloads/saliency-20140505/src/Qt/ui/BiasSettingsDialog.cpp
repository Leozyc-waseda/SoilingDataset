/****************************************************************************
** Form implementation generated from reading ui file 'Qt/BiasSettingsDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/BiasSettingsDialog.h"

#include <qvariant.h>
#include <qtabwidget.h>
#include <qwidget.h>
#include <qspinbox.h>
#include <qlabel.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Qt/ImageCanvas.h"
#include "Qt/BiasSettingsDialog.ui.h"

/*
 *  Constructs a BiasSettingsDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
BiasSettingsDialog::BiasSettingsDialog( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "BiasSettingsDialog" );
    setSizeGripEnabled( TRUE );
    setModal( FALSE );
    BiasSettingsDialogLayout = new QVBoxLayout( this, 11, 6, "BiasSettingsDialogLayout");

    tabDisp = new QTabWidget( this, "tabDisp" );

    tab = new QWidget( tabDisp, "tab" );

    spinBox43 = new QSpinBox( tab, "spinBox43" );
    spinBox43->setEnabled( FALSE );
    spinBox43->setGeometry( QRect( 117, 87, 42, 24 ) );

    textLabel10 = new QLabel( tab, "textLabel10" );
    textLabel10->setEnabled( FALSE );
    textLabel10->setGeometry( QRect( 117, 127, 87, 22 ) );

    imageCanvas50 = new ImageCanvas( tab, "imageCanvas50" );
    imageCanvas50->setEnabled( FALSE );
    imageCanvas50->setGeometry( QRect( 107, 157, 91, 31 ) );
    tabDisp->insertTab( tab, QString::fromLatin1("") );
    BiasSettingsDialogLayout->addWidget( tabDisp );

    layout4 = new QHBoxLayout( 0, 0, 6, "layout4");
    spacer2 = new QSpacerItem( 110, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout4->addItem( spacer2 );

    chkBoxShowRaw = new QCheckBox( this, "chkBoxShowRaw" );
    layout4->addWidget( chkBoxShowRaw );

    chkBoxResizeToSLevel = new QCheckBox( this, "chkBoxResizeToSLevel" );
    layout4->addWidget( chkBoxResizeToSLevel );

    updateValButton = new QPushButton( this, "updateValButton" );
    layout4->addWidget( updateValButton );

    buttonOk = new QPushButton( this, "buttonOk" );
    buttonOk->setAutoDefault( TRUE );
    buttonOk->setDefault( TRUE );
    layout4->addWidget( buttonOk );

    buttonCancel = new QPushButton( this, "buttonCancel" );
    buttonCancel->setAutoDefault( TRUE );
    layout4->addWidget( buttonCancel );
    BiasSettingsDialogLayout->addLayout( layout4 );
    languageChange();
    resize( QSize(729, 621).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( buttonOk, SIGNAL( clicked() ), this, SLOT( accept() ) );
    connect( buttonCancel, SIGNAL( clicked() ), this, SLOT( reject() ) );
    connect( updateValButton, SIGNAL( clicked() ), this, SLOT( update() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
BiasSettingsDialog::~BiasSettingsDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void BiasSettingsDialog::languageChange()
{
    setCaption( tr( "Bias Settings" ) );
    textLabel10->setText( tr( "textLabel10" ) );
    tabDisp->changeTab( tab, tr( "Tab 2" ) );
    chkBoxShowRaw->setText( tr( "Show RawCSMap" ) );
    chkBoxResizeToSLevel->setText( tr( "Resize to SLevel" ) );
    updateValButton->setText( tr( "Update" ) );
    buttonOk->setText( tr( "&OK" ) );
    buttonOk->setAccel( QKeySequence( QString::null ) );
    buttonCancel->setText( tr( "&Cancel" ) );
    buttonCancel->setAccel( QKeySequence( QString::null ) );
}

