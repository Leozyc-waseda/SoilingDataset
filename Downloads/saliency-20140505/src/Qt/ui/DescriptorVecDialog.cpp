/****************************************************************************
** Form implementation generated from reading ui file 'Qt/DescriptorVecDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/DescriptorVecDialog.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qtabwidget.h>
#include <qwidget.h>
#include <qtable.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Qt/ImageCanvas.h"
#include "Qt/DescriptorVecDialog.ui.h"

/*
 *  Constructs a DescriptorVecDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
DescriptorVecDialog::DescriptorVecDialog( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
        setName( "DescriptorVecDialog" );
    setSizeGripEnabled( TRUE );
    DescriptorVecDialogLayout = new QVBoxLayout( this, 11, 6, "DescriptorVecDialogLayout");

    FVtab = new QTabWidget( this, "FVtab" );

    tab = new QWidget( FVtab, "tab" );
    tabLayout = new QHBoxLayout( tab, 11, 6, "tabLayout");

    imgDisp = new ImageCanvas( tab, "imgDisp" );
    tabLayout->addWidget( imgDisp );
    FVtab->insertTab( tab, QString::fromLatin1("") );

    TabPage = new QWidget( FVtab, "TabPage" );
    TabPageLayout = new QVBoxLayout( TabPage, 11, 6, "TabPageLayout");

    histDisp = new ImageCanvas( TabPage, "histDisp" );
    TabPageLayout->addWidget( histDisp );
    FVtab->insertTab( TabPage, QString::fromLatin1("") );

    TabPage_2 = new QWidget( FVtab, "TabPage_2" );
    TabPageLayout_2 = new QHBoxLayout( TabPage_2, 11, 6, "TabPageLayout_2");

    FVtable = new QTable( TabPage_2, "FVtable" );
    FVtable->setNumRows( 0 );
    FVtable->setNumCols( 0 );
    TabPageLayout_2->addWidget( FVtable );
    FVtab->insertTab( TabPage_2, QString::fromLatin1("") );
    DescriptorVecDialogLayout->addWidget( FVtab );

    layout3 = new QHBoxLayout( 0, 0, 6, "layout3");
    spacer3 = new QSpacerItem( 21, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout3->addItem( spacer3 );

    buttonOk = new QPushButton( this, "buttonOk" );
    buttonOk->setAutoDefault( TRUE );
    buttonOk->setDefault( TRUE );
    layout3->addWidget( buttonOk );

    buttonCancel = new QPushButton( this, "buttonCancel" );
    buttonCancel->setAutoDefault( TRUE );
    layout3->addWidget( buttonCancel );
    DescriptorVecDialogLayout->addLayout( layout3 );
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
DescriptorVecDialog::~DescriptorVecDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void DescriptorVecDialog::languageChange()
{
    setCaption( tr( "Descriptor Vector Display" ) );
    FVtab->changeTab( tab, tr( "Image" ) );
    FVtab->changeTab( TabPage, tr( "Histogram" ) );
    FVtab->changeTab( TabPage_2, tr( "Feature Vector" ) );
    buttonOk->setText( tr( "&OK" ) );
    buttonOk->setAccel( QKeySequence( QString::null ) );
    buttonCancel->setText( tr( "&Cancel" ) );
    buttonCancel->setAccel( QKeySequence( QString::null ) );
}

