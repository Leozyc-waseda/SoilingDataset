/*! @file Qt/sscControlQt.cpp Qt interface for ssc control */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/sscControlQt.cpp $
// $Id: sscControlQt.cpp 5957 2005-11-16 18:10:52Z rjpeters $

#include <qapplication.h>
#include "Qt/ui/SSCMainForm.h"

int main( int argc, char ** argv )
{
  // instantiate a model manager:
  ModelManager manager("sscControlQt");

  // instantiate our various ModelComponents:
  nub::soft_ref<SSC> ssc(new SSC(manager));
  manager.addSubComponent(ssc);

  // Let's get going:
  manager.start();

  // get the Qt form up and going:
  QApplication a(argc, argv);
  SSCMainForm *w = new SSCMainForm;
  w->init(&manager, ssc);
  w->show();
  a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));

  // run the form until the user quits it:
  int retval = a.exec();

  // close down the manager:
  manager.stop();

  return retval;
}
