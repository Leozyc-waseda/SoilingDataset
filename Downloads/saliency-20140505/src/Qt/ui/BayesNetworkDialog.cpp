/****************************************************************************
** Form implementation generated from reading ui file 'Qt/BayesNetworkDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/BayesNetworkDialog.h"

#include <qvariant.h>
#include <qtabwidget.h>
#include <qwidget.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Qt/BayesNetworkDialog.ui.h"

/*
 *  Constructs a BayesNetworkDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
BayesNetworkDialog::BayesNetworkDialog( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "BayesNetworkDialog" );
    setSizeGripEnabled( TRUE );
    BayesNetworkDialogLayout = new QVBoxLayout( this, 11, 6, "BayesNetworkDialogLayout");

    tabWidget = new QTabWidget( this, "tabWidget" );

    tab = new QWidget( tabWidget, "tab" );
    tabWidget->insertTab( tab, QString::fromLatin1("") );
    BayesNetworkDialogLayout->addWidget( tabWidget );

    layout2 = new QHBoxLayout( 0, 0, 6, "layout2");
    spacer2 = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout2->addItem( spacer2 );

    buttonOk = new QPushButton( this, "buttonOk" );
    buttonOk->setAutoDefault( TRUE );
    buttonOk->setDefault( TRUE );
    layout2->addWidget( buttonOk );

    buttonCancel = new QPushButton( this, "buttonCancel" );
    buttonCancel->setAutoDefault( TRUE );
    layout2->addWidget( buttonCancel );
    BayesNetworkDialogLayout->addLayout( layout2 );
    languageChange();
    resize( QSize(567, 587).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( buttonOk, SIGNAL( clicked() ), this, SLOT( accept() ) );
    connect( buttonCancel, SIGNAL( clicked() ), this, SLOT( reject() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
BayesNetworkDialog::~BayesNetworkDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void BayesNetworkDialog::languageChange()
{
    setCaption( tr( "MyDialog" ) );
    tabWidget->changeTab( tab, tr( "BayesNet" ) );
    buttonOk->setText( tr( "&OK" ) );
    buttonOk->setAccel( QKeySequence( QString::null ) );
    buttonCancel->setText( tr( "&Cancel" ) );
    buttonCancel->setAccel( QKeySequence( QString::null ) );
}

