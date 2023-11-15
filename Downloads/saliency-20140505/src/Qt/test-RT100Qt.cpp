/*! @file Qt/test-RT100Qt.cpp rt100 QT arm control : */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-RT100Qt.cpp $
// $Id: test-RT100Qt.cpp 8267 2007-04-18 18:24:24Z rjpeters $

#include "Qt/ui/rt100controlForm.h"
#include "Component/ModelManager.H"
#include "Devices/rt100.H"
#include <qapplication.h>

int main( int argc, char ** argv )
{
  ModelManager manager("RT100 robot control");

  nub::soft_ref<RT100> rt100(new RT100(manager));
  manager.addSubComponent(rt100);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  manager.start();

  QApplication a( argc, argv );
  RT100ControlForm rt100Form;
  rt100Form.init(&manager, rt100);
  rt100Form.show();
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );

  int retval = a.exec();

  manager.stop();

  return retval;

}
