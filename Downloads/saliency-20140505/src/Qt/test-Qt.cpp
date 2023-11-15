/*! @file Qt/test-Qt.cpp basic Qt widget toy */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-Qt.cpp $
// $Id: test-Qt.cpp 5957 2005-11-16 18:10:52Z rjpeters $

#include <qapplication.h>
#include "Qt/ui/QtTestForm.h"
#include "Util/Types.H"
#include "Util/log.H"

int main( int argc, char ** argv )
{
  QApplication a( argc, argv );
  QtTestForm *w = new QtTestForm;
  w->show();
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );
  LINFO( "Now starting application test-Qt" );
  return a.exec();
}
