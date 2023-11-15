/*! @file Qt/test-ImageQt.cpp test Qt interface for Image reading/writing/display */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-ImageQt.cpp $
// $Id: test-ImageQt.cpp 5957 2005-11-16 18:10:52Z rjpeters $

#include <qapplication.h>
#include "Qt/ui/ImageQtMainForm.h"

int main( int argc, char ** argv )
{
  QApplication a( argc, argv );
  ImageQtMainForm *w = new ImageQtMainForm;
  w->show();
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );
  return a.exec();
}
