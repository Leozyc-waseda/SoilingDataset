/****************************************************************************
** Form implementation generated from reading ui file 'Qt/ModelManagerDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/ModelManagerDialog.h"

#include <qvariant.h>
#include <qlabel.h>
#include <qpushbutton.h>
#include <qheader.h>
#include <qlistview.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Qt/ModelManagerDialog.ui.h"

/*
 *  Constructs a ModelManagerDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ModelManagerDialog::ModelManagerDialog( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "ModelManagerDialog" );

    textLabel1 = new QLabel( this, "textLabel1" );
    textLabel1->setGeometry( QRect( 10, 10, 600, 130 ) );
    textLabel1->setAlignment( int( QLabel::WordBreak | QLabel::AlignVCenter ) );

    wizardButton = new QPushButton( this, "wizardButton" );
    wizardButton->setGeometry( QRect( 20, 540, 90, 31 ) );

    cancelButton = new QPushButton( this, "cancelButton" );
    cancelButton->setGeometry( QRect( 520, 540, 81, 31 ) );

    applyButton = new QPushButton( this, "applyButton" );
    applyButton->setGeometry( QRect( 430, 540, 70, 30 ) );
    applyButton->setDefault( TRUE );

    listview = new QListView( this, "listview" );
    listview->addColumn( tr( "Model Parameter" ) );
    listview->addColumn( tr( "Value" ) );
    listview->setGeometry( QRect( 10, 150, 600, 370 ) );
    languageChange();
    resize( QSize(624, 586).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( applyButton, SIGNAL( clicked() ), this, SLOT( handleApplyButton() ) );
    connect( listview, SIGNAL( itemRenamed(QListViewItem*,int,const QString&) ), this, SLOT( handleItemEdit(QListViewItem*) ) );
    connect( cancelButton, SIGNAL( clicked() ), this, SLOT( handleCancelButton() ) );
    connect( wizardButton, SIGNAL( clicked() ), this, SLOT( handleWizardButton() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
ModelManagerDialog::~ModelManagerDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void ModelManagerDialog::languageChange()
{
    setCaption( tr( "Model Configuration" ) );
    textLabel1->setText( tr( "Model parameters for this program are listed in the configuration box below.  Note that some of these components will require reconfiguration if you change its type, such as FrameGrabberType in FrameGrabberConfigurator.  When everything is configured, press <b>Apply</b>, or press <b>Cancel</b> to discard your changes.<br><br>If you are not sure how to set the parameters, you can try using a guided approach by pressing<b> Wizard...</b>" ) );
    wizardButton->setText( tr( "Wizard..." ) );
    cancelButton->setText( tr( "Cancel" ) );
    applyButton->setText( tr( "Apply" ) );
    listview->header()->setLabel( 0, tr( "Model Parameter" ) );
    listview->header()->setLabel( 1, tr( "Value" ) );
}

