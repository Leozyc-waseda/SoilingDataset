/****************************************************************************
** Form implementation generated from reading ui file 'Qt/ModelManagerWizard.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/ModelManagerWizard.h"

#include <qvariant.h>
#include <algorithm>
#include <qwhatsthis.h>
#include <qcheckbox.h>
#include <qlineedit.h>
#include <qerrormessage.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qlistbox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Util/StringConversions.H"
#include "Qt/ModelManagerWizard.ui.h"

/*
 *  Constructs a ModelManagerWizard as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ModelManagerWizard::ModelManagerWizard( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "ModelManagerWizard" );
    setPaletteBackgroundColor( QColor( 234, 233, 232 ) );

    cancelButton = new QPushButton( this, "cancelButton" );
    cancelButton->setGeometry( QRect( 11, 498, 80, 25 ) );

    nextButton = new QPushButton( this, "nextButton" );
    nextButton->setGeometry( QRect( 600, 500, 80, 25 ) );

    backButton = new QPushButton( this, "backButton" );
    backButton->setGeometry( QRect( 500, 500, 80, 25 ) );

    finishButton = new QPushButton( this, "finishButton" );
    finishButton->setGeometry( QRect( 700, 500, 80, 25 ) );

    textLabel2 = new QLabel( this, "textLabel2" );
    textLabel2->setGeometry( QRect( 280, 460, 450, 20 ) );
    QFont textLabel2_font(  textLabel2->font() );
    textLabel2_font.setFamily( "Helvetica" );
    textLabel2->setFont( textLabel2_font );

    listbox = new QListBox( this, "listbox" );
    listbox->setGeometry( QRect( 11, 11, 240, 481 ) );
    languageChange();
    resize( QSize(791, 534).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( listbox, SIGNAL( currentChanged(QListBoxItem*) ), this, SLOT( showFrame(QListBoxItem*) ) );
    connect( cancelButton, SIGNAL( clicked() ), this, SLOT( handleCancelButton() ) );
    connect( backButton, SIGNAL( clicked() ), this, SLOT( handleBackButton() ) );
    connect( nextButton, SIGNAL( clicked() ), this, SLOT( handleNextButton() ) );
    connect( finishButton, SIGNAL( clicked() ), this, SLOT( handleFinishButton() ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
ModelManagerWizard::~ModelManagerWizard()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void ModelManagerWizard::languageChange()
{
    setCaption( tr( "Model Manager Configuration Wizard" ) );
    cancelButton->setText( tr( "Cancel" ) );
    nextButton->setText( tr( "Next >" ) );
    backButton->setText( tr( "< Back" ) );
    finishButton->setText( tr( "Finish" ) );
    textLabel2->setText( tr( "For more information about an option, right click it and select <b>What's this?</b>" ) );
}

