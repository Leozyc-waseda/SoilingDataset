/*!@file Corba/Objects/EchoImageServer.C a test program to send an image using corba:
  This is the server class that echos the image  */

//////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Objects/EchoImageServer.C $
// $Id: EchoImageServer.C 6795 2006-06-29 20:45:32Z rjpeters $
//

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"
#include "SIFT/ScaleSpace.H"
#include "EchoImageServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/ImageOrbUtil.H"
#include "Corba/CorbaUtil.H"
#include "Corba/Objects/EchoImageServer.H"

CORBA::ORB_var orb;
CosNaming::Name objectName;

bool Debug = false;

//! Signal handler (e.g., for control-C)
void terminate(int s)
{
  LERROR("*** INTERRUPT ***");
  unbindObject(orb, "saliency", "EchoImage", objectName);
  orb->shutdown(0);
}

void EchoImageServer_i::shutdown()
{
  // Shutdown the ORB
  unbindObject(orb, "saliency", "EchoImage", objectName);
  orb->shutdown(0);
}


EchoImageServer_i::EchoImageServer_i()
{
}


EchoImageServer_i::~EchoImageServer_i()
{

}

ImageOrb* EchoImageServer_i::echo(const ImageOrb& img)
{
  LINFO("Got Image");
  Image< PixRGB<byte> > image;
  orb2Image(img, image);

  LINFO("Echoing image back");
  return image2Orb(image);
}


//start the class server
int main(int argc, char **argv)
{

  MYLOGVERB = LOG_INFO;

  // Instantiate a ModelManager:
  ModelManager manager("EchoImage Corba Object");

  if (manager.parseCommandLine((const int)argc, (const char**)argv, "", 0, 0) == false)
    return(1);

  if (manager.debugMode()){
    Debug = true;
  } else {
    LINFO("Running as a daemon. Set --debug to see any errors.");
    //Become a daemon
    // fork off the parent process
    pid_t pid = fork();
    if (pid < 0)
      LFATAL("Can not fork");

    if (pid > 0)
      exit(0); //exit the parent process

    // Change the file mask
    umask(0);

    //Create a new system id so that the kernel wont think we are an orphan.
    pid_t sid = setsid();
    if (sid < 0)
      LFATAL("Can not become independent");

    fclose(stdin);
    fclose(stdout);
    fclose(stderr);

  }
  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);


  //Create the object and run in
  orb = CORBA::ORB_init(argc, argv);

  CORBA::Object_var obj = orb->resolve_initial_references("RootPOA");
  PortableServer::POA_var poa = PortableServer::POA::_narrow(obj);

  EchoImageServer_i* myobj = new EchoImageServer_i();


  PortableServer::ObjectId_var objID = poa->activate_object(myobj);

  //get a ref string
  obj = myobj->_this();
  CORBA::String_var sior(orb->object_to_string(obj));
  std::cerr << "'" << (char*)sior << "'" << "\n";

  if( !bindObjectToName(orb, obj, "saliency", "EchoImage", "EchoImage", objectName) )
    return 1;
  myobj->_remove_ref();

  PortableServer::POAManager_var pman = poa->the_POAManager();
  pman->activate();

  try
  {
    manager.start();
    //run the object untill shutdown or killed
    orb->run();
  } catch (...) {
    LINFO("Error starting server");
  }

  LINFO("Shutting down");
  unbindObject(orb, "saliency", "EchoImage", objectName);

  manager.stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
