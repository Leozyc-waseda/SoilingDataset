/*! @file Qt/test-HeadQt.cpp thest the robot head */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-HeadQt.cpp $
// $Id: test-HeadQt.cpp 8353 2007-05-06 16:31:00Z lior $

#include <qapplication.h>
#include "Qt/ui/RobotHeadForm.h"
#include "Component/ModelManager.H"
#include "Devices/BeoHead.H"
#include "Devices/DeviceOpts.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Transport/TransportOpts.H"
#include "Component/ParamMap.H"
#include "Component/GlobalOpts.H"
#include "Simulation/SimulationOpts.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "rutz/shared_ptr.h"
#include "rutz/trace.h"
#include "QtUtil/Util.H"

int main( int argc, const char ** argv )
{

  ModelManager mgr("Robot Head");

  nub::soft_ref<BeoHead> beoHead(new BeoHead(mgr));
  mgr.addSubComponent(beoHead);

  mgr.exportOptions(MC_RECURSE);

 /* mgr.setOptionValString(&OPT_FrameGrabberStreaming, "false");
  mgr.setOptionValString(&OPT_FrameGrabberType, "V4L");
  mgr.setOptionValString(&OPT_FrameGrabberChannel, "1");
  mgr.setOptionValString(&OPT_FrameGrabberHue, "0");
  mgr.setOptionValString(&OPT_FrameGrabberContrast, "16384");
  mgr.setOptionValString(&OPT_FrameGrabberDims, "320x240");

  if (mgr.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  nub::soft_ref<FrameIstream> gb;
  gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
        "--fg-type=XX command-line option for this program "
        "to be useful");*/



  mgr.start();

  QApplication a( argc, argv2qt(argc, argv) );
  RobotHeadForm *w = new RobotHeadForm;
  w->init(mgr, beoHead);
  w->show();
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );
  return a.exec();
}
