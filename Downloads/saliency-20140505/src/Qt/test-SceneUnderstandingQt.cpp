/*! @file Qt/test-SceneUnderstandingQt.cpp Image interface for scenes understanding */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-SceneUnderstandingQt.cpp $
// $Id: test-SceneUnderstandingQt.cpp 10982 2009-03-05 05:11:22Z itti $

#include <qapplication.h>
#include "Qt/ui/SceneUnderstandingForm.h"
#include "Component/ModelManager.H"
#include "Component/ParamMap.H"
#include "Component/GlobalOpts.H"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortexConfigurator.H"
#include "Neuro/NeuroOpts.H"
#include "Simulation/SimulationOpts.H"
#include "Simulation/SimEventQueueConfigurator.H"

#include "QtUtil/Util.H"

int main( int argc, const char ** argv )
{

  ModelManager mgr("Bias Image");

  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(mgr));
  mgr.addSubComponent(seqc);

  //our brain
  nub::ref<StdBrain>  brain(new StdBrain(mgr));
  mgr.addSubComponent(brain);

  mgr.exportOptions(MC_RECURSE);

  mgr.setOptionValString(&OPT_RawVisualCortexChans, "IOC");
  //mgr.setOptionValString(&OPT_RawVisualCortexChans, "I");
  //mgr.setOptionValString(&OPT_RawVisualCortexChans, "GNO");
  //mgr.setOptionValString(&OPT_RawVisualCortexChans, "N");
  //manager.setOptionValString(&OPT_UseOlderVersion, "false");
  // set the FOA and fovea radii
  mgr.setOptionValString(&OPT_SaliencyMapType, "Fast");
  mgr.setOptionValString(&OPT_SMfastInputCoeff, "1");

  mgr.setOptionValString(&OPT_WinnerTakeAllType, "Fast");
  mgr.setOptionValString(&OPT_SimulationTimeStep, "0.2");

  mgr.setModelParamVal("FOAradius", 40, MC_RECURSE);
  mgr.setModelParamVal("FoveaRadius", 40, MC_RECURSE);


  mgr.setOptionValString(&OPT_IORtype, "Disc");

  if (mgr.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  mgr.start();

  QApplication a( argc, argv2qt(argc, argv) );
  SceneUnderstandingForm *w = new SceneUnderstandingForm;
  w->init(mgr);
  w->show();
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );
  return a.exec();
}
