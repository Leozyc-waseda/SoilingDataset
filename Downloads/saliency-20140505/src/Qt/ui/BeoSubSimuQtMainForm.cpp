/****************************************************************************
** Form implementation generated from reading ui file 'Qt/BeoSubSimuQtMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/BeoSubSimuQtMainForm.h"

#include <qvariant.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "Raster/Raster.H"
#include "Qt/simulation.h"
#include "Qt/BeoSubSimuQtMainForm.ui.h"

/*
 *  Constructs a BeoSubSimuQtMainForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
BeoSubSimuQtMainForm::BeoSubSimuQtMainForm( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
        setName( "BeoSubSimuQtMainForm" );

    simulation1 = new Simulation( this, "simulation1" );
    simulation1->setGeometry( QRect( 60, 100, 481, 331 ) );
    languageChange();
    resize( QSize(600, 480).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
    init();
}

/*
 *  Destroys the object and frees any allocated resources
 */
BeoSubSimuQtMainForm::~BeoSubSimuQtMainForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void BeoSubSimuQtMainForm::languageChange()
{
    setCaption( tr( "Simulation" ) );
}

