/*!@file Corba/Objects/SceneRecServer.C control a SceneRec via corba */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Objects/SceneRecServer.C $
// $Id: SceneRecServer.C 9412 2008-03-10 23:10:15Z farhan $
//

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Util/Assert.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Component/ModelManager.H"
#include "Corba/Objects/SceneRecServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/Objects/SceneRecServer.H"
#include "Corba/CorbaUtil.H"

CORBA::ORB_var orb;
CosNaming::Name objectName;

bool Debug = false;

//! Signal handler (e.g., for control-C)
void terminate(int s)
{
        LERROR("*** INTERRUPT ***");
        unbindObject(orb, "saliency", "SceneRec", objectName);
        orb->shutdown(0);
}

void SceneRecServer_i::shutdown() {
        // Shutdown the ORB
        unbindObject(orb, "saliency", "SceneRec", objectName);
        orb->shutdown(0);
}


SceneRecServer_i::SceneRecServer_i(OptionManager &mgr) :
        SceneRec(mgr)
{
}


SceneRecServer_i::~SceneRecServer_i() {

}


void SceneRecServer_i::newInput(const ImageOrb &img){
        Image<PixRGB<byte> > image;
        orb2Image(img, image);
        SceneRec::newInput(image);


}

short SceneRecServer_i::outputReady(){

        return SceneRec::outputReady();
}

short SceneRecServer_i::getLandmarkLoc(Point2DOrb &loc){

        Point2D<int> location;

        short leg = SceneRec::getLandmarkLoc(location);
        loc.i = location.i;
        loc.j = location.j;

                  return leg;
}

void SceneRecServer_i::trainFeature(const ImageOrb &img, const Point2DOrb &loc,
                const DimsOrb &window, const short leg){

        Image<PixRGB<byte> > image;
        orb2Image(img, image);

        SceneRec::trainFeature(image, Point2D<int>(loc.i, loc.j), Dims(window.ww, window.hh), leg);
}


//start the class server
int main(int argc, char **argv){

        MYLOGVERB = LOG_INFO;

        // Instantiate a ModelManager:
        ModelManager manager("SceneRecServer Corba Object");
        nub::ref<SceneRecServer_i> sceneRec(new SceneRecServer_i(manager));
        manager.addSubComponent(sceneRec);

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

        SceneRecServer_i* myobj = sceneRec.get();

        PortableServer::ObjectId_var objID = poa->activate_object(myobj);

        //get a ref string
        obj = myobj->_this();
        CORBA::String_var sior(orb->object_to_string(obj));
        std::cerr << "'" << (char*)sior << "'" << "\n";

        if( !bindObjectToName(orb, obj, "saliency", "SceneRec", "SceneRecServer", objectName) )
                return 1;
        myobj->_remove_ref();

        PortableServer::POAManager_var pman = poa->the_POAManager();
        pman->activate();


        try {
                manager.start();
                //run the object untill shutdown or killed
                orb->run();
        } catch (...) {
                LINFO("Error starting server");
        }


        LINFO("Shutting down");
        unbindObject(orb, "saliency", "SceneRec", objectName);

        manager.stop();

        return 0;
}

