#include "Qt/SeaBee3GUICommunicator.H"
#include "Qt/ui/SeaBee3GUI.h"
#include "Component/ModelManager.H"
#include <qapplication.h>
#include <Ice/Ice.h>
#include <Ice/Service.h>
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Image/CutPaste.H"
#include "Robots/SeaBeeIII/XBox360RemoteControlI.H"

class SeaBee3GUIService : public Ice::Service {
  protected:
    virtual bool start(int, char* argv[]);
    virtual bool stop() {
      if (itsMgr)
        delete itsMgr;
      return true;
    }

  private:
    Ice::ObjectAdapterPtr itsAdapter;
    ModelManager *itsMgr;
};

bool SeaBee3GUIService::start( int argc, char ** argv )
{
  QApplication a( argc, argv );

  itsMgr = new ModelManager("SeaBee3 GUI");

  //Create the adapter

  char adapterStr[255];
  LINFO("Creating Adapter");
  sprintf(adapterStr, "default -p %i", 12345);
  itsAdapter = communicator()->createObjectAdapterWithEndpoints("RobotBrainPort",
                                                                adapterStr);

  LINFO("Initializing Main Form");
  SeaBee3MainDisplayForm *mainForm = new SeaBee3MainDisplayForm();
  mainForm->init(itsMgr);

  LINFO("Creating Communicator");
  nub::soft_ref<SeaBee3GUICommunicator> GUIComm(new SeaBee3GUICommunicator(*itsMgr, "SeaBee3GUICommunicator", "SeaBee3GUICommunicator"));
  itsMgr->addSubComponent(GUIComm);

  LINFO("Registering GUI With GUICommunicator");
  GUIComm->registerGUI(mainForm);

  LINFO("Registering Communicator with GUI");
  mainForm->registerCommunicator(GUIComm);

  LINFO("Starting Up GUI Comm");
  GUIComm->init(communicator(), itsAdapter);

  LINFO("Starting XBox360RemoteControl");
  nub::soft_ref<XBox360RemoteControlI> xbox360remote(new XBox360RemoteControlI(0,*itsMgr, "XBox360RemoteControl", "XBox360RemoteControl"));

  LINFO("XBox360RemoteControl Created");
  itsMgr->addSubComponent(xbox360remote);

  LINFO("XBox360RemoteControl Added As Sub Component");
  xbox360remote->init(communicator(), itsAdapter);

  LINFO("XBox360RemoteControl Inited");


  // Parse command-line:
  if (itsMgr->parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  itsAdapter->activate();

  LINFO("Starting Manager");
  itsMgr->start();

  LINFO("Showing Main Form");
  mainForm->show();

  LINFO("Waiting for Main Form to exit");
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );

  int retval = a.exec();

  itsMgr->stop();

  return retval;

}

// ######################################################################
int main(int argc, char** argv) {

  SeaBee3GUIService svc;
  return svc.main(argc, argv);
}

