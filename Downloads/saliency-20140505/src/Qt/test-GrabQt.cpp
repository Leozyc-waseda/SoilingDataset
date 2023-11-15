/*! @file Qt/test-GrabQt.cpp basic Qt framegrabber toy
(take a picture of yourself with it!) */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-GrabQt.cpp $
// $Id: test-GrabQt.cpp 5957 2005-11-16 18:10:52Z rjpeters $

#include "Qt/ui/GrabQtMainForm.h"
#include "QtUtil/Util.H"

#include <qapplication.h>

int main( int argc, const char ** argv )
{
  QApplication a( argc, argv2qt( argc, argv ) );
  GrabQtMainForm gq;
  gq.init( argc, argv );
  gq.show();
  gq.move( 10, 10 );
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );
  return a.exec();
}
