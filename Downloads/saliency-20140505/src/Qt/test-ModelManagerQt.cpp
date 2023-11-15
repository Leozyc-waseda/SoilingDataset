/*! @file Qt/test-ModelManagerQt.cpp test Qt interface for a more interactive alternative to command line configs */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-ModelManagerQt.cpp $
// $Id: test-ModelManagerQt.cpp 10993 2009-03-06 06:05:33Z itti $

#include "Component/ModelManager.H"
#include "Qt/ui/ModelManagerControl.h"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortexConfigurator.H"
#include "QtUtil/Util.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Component/JobServerConfigurator.H"
#include "Media/SimFrameSeries.H"
#include "Neuro/NeuroOpts.H"

#include <qapplication.h>
#include <qthread.h>
#include <unistd.h>

class ModelRun : public QThread {
public:
  ModelRun(ModelManager *mgr_, nub::ref<SimEventQueueConfigurator> seqc_, bool *dorun_) :
    QThread(), mgr(mgr_), seqc(seqc_), dorun(dorun_)
  { }

  virtual void run() {
    while(true) {
      // wait until ready to start
      if (*dorun == false) { usleep(50000); continue; }

      // let's do it!
      mgr->start();
      nub::ref<SimEventQueue> q = seqc->getQ();

      // run until all data has been processed:
      SimStatus status = SIM_CONTINUE;
      while (status == SIM_CONTINUE && *dorun == true) status = q->evolve();

      // print final memory allocation stats
      LINFO("Simulation terminated.");

      // stop all our ModelComponents
      mgr->stop();
    }
  }

private:
  ModelManager *mgr;
  nub::ref<SimEventQueueConfigurator> seqc;
  bool *dorun;
};

int main( int argc, const char ** argv )
{
  // instantiate a model manager:
  ModelManager mgr( "model manager" );

  // instantiate our various ModelComponents:
  nub::ref<JobServerConfigurator> jsc(new JobServerConfigurator(mgr));
  mgr.addSubComponent(jsc);

  nub::ref<SimEventQueueConfigurator> seqc(new SimEventQueueConfigurator(mgr));
  mgr.addSubComponent(seqc);

  // NOTE: make sure you register your OutputFrameSeries with the
  // manager before you do your InputFrameSeries, to ensure that
  // outputs for the current frame get saved before the next input
  // frame is loaded.
  nub::ref<SimOutputFrameSeries> ofs(new SimOutputFrameSeries(mgr));
  mgr.addSubComponent(ofs);

  nub::ref<SimInputFrameSeries> ifs(new SimInputFrameSeries(mgr));
  mgr.addSubComponent(ifs);

  nub::ref<StdBrain> brain(new StdBrain(mgr));
  mgr.addSubComponent(brain);

  // Request a bunch of option aliases (shortcuts to lists of options):
  REQUEST_OPTIONALIAS_NEURO(mgr);

  // Parse command-line:
  if (mgr.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  // NOTE: change this so that it brings up alert box and config dialog

  // get our thread going:
  bool dorun = false;
  ModelRun mr(&mgr, seqc, &dorun);
  mr.start();

  // show the configuration dialog
  QApplication a( argc, argv2qt( argc, argv ) );
  ModelManagerControl mmc;
  mmc.init( mgr, &dorun );
  mmc.show();
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );
  a.exec();

  return 0;
}
