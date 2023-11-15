/*!@file Corba/Objects/BotControlServer.C control a robot via corba */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Corba/Objects/BotControlServer.C $
// $Id: BotControlServer.C 10794 2009-02-08 06:21:09Z itti $
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
#include "Corba/Objects/BotControlServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/Objects/BotControlServer.H"
#include "Corba/CorbaUtil.H"

CORBA::ORB_var orb;
CosNaming::Name objectName;

bool Debug = false;

//! Signal handler (e.g., for control-C)
void terminate(int s)
{
        LERROR("*** INTERRUPT ***");
        unbindObject(orb, "saliency", "BotControllers", objectName);
        orb->shutdown(0);
}

void BotControlServer_i::shutdown() {
        // Shutdown the ORB
        unbindObject(orb, "saliency", "BotControllers", objectName);
        orb->shutdown(0);
}


BotControlServer_i::BotControlServer_i(OptionManager &mgr, bool debug) :
        BotControl(mgr, "Bot control", "BotControl", BotControl::SIMULATION)
{
}


BotControlServer_i::~BotControlServer_i() {

}



void  BotControlServer_i::init(){
        BotControl::init();

}


float BotControlServer_i::getSpeed(){
        return BotControl::getSpeed();
}


short BotControlServer_i::setSpeed(const float speed){
        return BotControl::setSpeed(speed);
}


float BotControlServer_i::getSteering(){
        return BotControl::getSteering();
}


short BotControlServer_i::setSteering(const float steeringPos){
        return BotControl::setSteering(steeringPos);
}


ImageOrb* BotControlServer_i::getImageSensor(const short i){

        Image<PixRGB<byte> > image = BotControl::getImageSensor(i);

        return image2Orb(image);
}


void BotControlServer_i::getImageSensorDims(short &w, short &h, const short i) {

        BotControl::getImageSensorDims(w, h, i);
}

void BotControlServer_i::setInfo(const char *info, const Point2DOrb& trackLoc,
                const Point2DOrb& recLoc){
        BotControl::setInfo(info, Point2D<int>(trackLoc.i, trackLoc.j), Point2D<int>(recLoc.i, recLoc.j));
}

short BotControlServer_i::getUserInput(Point2DOrb &loc){
                Point2D<int> location;

                short key = BotControl::getUserInput(location);
                loc.i = location.i;
                loc.j = location.j;

                return key;
}

//start the class server
int main(int argc, char **argv){

        MYLOGVERB = LOG_INFO;

        // Instantiate a ModelManager:
        ModelManager manager("BotControlServer Corba Object");

        nub::ref<BotControlServer_i> botControl(new BotControlServer_i(manager, false));
        manager.addSubComponent(botControl);

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

        BotControlServer_i* myobj = botControl.get();


        PortableServer::ObjectId_var objID = poa->activate_object(myobj);

        //get a ref string
        obj = myobj->_this();
        CORBA::String_var sior(orb->object_to_string(obj));
        std::cerr << "'" << (char*)sior << "'" << "\n";

        if( !bindObjectToName(orb, obj, "saliency", "BotControllers", "BotControlServer", objectName) )
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
        unbindObject(orb, "saliency", "BotControllers", objectName);

        manager.stop();

        return 0;
}

