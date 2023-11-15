/*!@file SalGlassesServer.cc control salGlasses via corba */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SalGlasses/SalGlassesServer/SalGlassesServer.cc $
// $Id: SalGlassesServer.cc 9108 2007-12-30 06:14:30Z rjpeters $
//

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <SalGlasses.hh>
#include "corbaUtil.h"
#include "SalGlassesServer.h"
#include "capture.h"
#include "serial.h"

#define OBJECT_NS "saliency"
#define OBJECT_NAME "SalGlasses"
CORBA::ORB_var orb;
CosNaming::Name objectName;
int serialFd = -1;

// ######################################################################
// Thunk to convert from env_size_t to size_t
static void* malloc_thunk(env_size_t n)
{
        return malloc(n);
}


// ######################################################################
static void assert_handler(const char* what, int custom_msg,
                           const char* where, int line_no)
        __attribute__((noreturn));

static void assert_handler(const char* what, int custom_msg,
                           const char* where, int line_no)
{
        if (custom_msg)
                fprintf(stderr, "Assertion failed (%s:%d):\n\t%s\n\n",
                        where, line_no, what);
        else
                fprintf(stderr, "Assertion failed (%s:%d):\n\texpected '%s'\n\n",
                        where, line_no, what);
        abort();
}

// ######################################################################
struct status_data
{
        int frame_number;
};

static void print_chan_status(void* userdata,
    const char* tagName,
    const struct env_image* img)
{
  struct status_data* p = (struct status_data*) userdata;

  if (env_img_initialized(img))
  {
    intg32 mi, ma;
    env_c_get_min_max(env_img_pixels(img), env_img_size(img),
        &mi, &ma);
    fprintf(stderr,
        "frame %06d channel status: "
        "%20s: range [%ld .. %ld]\n",
        p->frame_number, tagName, (long) mi, (long) ma);
  }
}

//! Signal handler (e.g., for control-C)
void terminate(int s)
{
        printf("*** INTERRUPT ***\n");
        unbindObject(orb, OBJECT_NS, OBJECT_NAME, objectName);
        orb->shutdown(0);
}

void SalGlassesServer::shutdown() {
        // Shutdown the ORB
        unbindObject(orb, OBJECT_NS, OBJECT_NAME, objectName);
        orb->shutdown(0);
}


SalGlassesServer::SalGlassesServer(int debug) :
  currentSpeed(0),
  currentSteering(0),
  itsDebug(debug)
{

  //start the serial device
 // itsSerialFd = openPort("/dev/ttyS2");

  //the envision object
  env_params_set_defaults(&itsEnvp);

  itsEnvp.maxnorm_type = ENV_VCXNORM_MAXNORM;
  itsEnvp.scale_bits = 16;
  env_assert_set_handler(&assert_handler);
  env_allocation_init(&malloc_thunk, &free);

  env_visual_cortex_init(&itsIvc, &itsEnvp);


}


SalGlassesServer::~SalGlassesServer() {

 // closePort(itsSerialFd);
  env_visual_cortex_destroy(&itsIvc);
  env_allocation_cleanup();
}



void  SalGlassesServer::init(){
}


void SalGlassesServer::getImageSensorDims(short &w, short &h, const short i) {
  w = 10; h = 10;
}

ImageOrb* SalGlassesServer::getImageSensor(const short i){


  frame* f = get_frame();
  int size = f->width*f->height*3;

  ImageOrb *imageOrb = new ImageOrb;
  imageOrb->width = f->width;
  imageOrb->height = f->height;
  imageOrb->pix_size = 3;

  CORBA::Octet *dat = new CORBA::Octet[size];

  memcpy(dat, f->data, size);
  imageOrb->data.replace(size, size, dat, 1); //release the data when delete the sequance

  return imageOrb;
}

ImageOrb* SalGlassesServer::getSaliencyMap(const short i){


  frame* f = get_frame();
  int size = f->width*f->height*3;

  struct env_image ivcout = env_img_initializer;
  struct env_image intens = env_img_initializer;
  struct env_image color = env_img_initializer;
  struct env_image ori = env_img_initializer;
#ifdef ENV_WITH_DYNAMIC_CHANNELS
  struct env_image flicker = env_img_initializer;
  struct env_image motion = env_img_initializer;
#endif

  struct status_data userdata;
  userdata.frame_number = 0;

  struct env_dims indims;
  struct env_rgb_pixel *input;
  input  = (env_rgb_pixel*)f->data;
  indims.w = f->width;
  indims.h = f->height;

  env_mt_visual_cortex_input(0, //not multithreaded
      &itsIvc, &itsEnvp,
      "visualcortex",
      input, 0, indims,
      &print_chan_status,
      &userdata,
      &ivcout,
      &intens, &color, &ori
#ifdef ENV_WITH_DYNAMIC_CHANNELS
      , &flicker, &motion
#endif
      );

  env_visual_cortex_rescale_ranges(
      &ivcout, &intens, &color, &ori
#ifdef ENV_WITH_DYNAMIC_CHANNELS
      , &flicker, &motion
#endif
      );



  ImageOrb *imageOrb = new ImageOrb;
  imageOrb->width = ivcout.dims.w;
  imageOrb->height = ivcout.dims.h;
  imageOrb->pix_size = 1;

  CORBA::Octet *dat = new CORBA::Octet[size];

  memcpy(dat, f->data, size);
  imageOrb->data.replace(size, size, dat, 1); //release the data when delete the sequance

  return imageOrb;
}


void SalGlassesServer::getWinner(short &x, short &y) {
  x = 10; y = 10;
}

short SalGlassesServer::getSensorValue(const short i) {
  return -1;
}



//start the class server
int main(int argc, char **argv){

  printf("Starting server\n");

  try {
  //Create the object and run in
  orb = CORBA::ORB_init(argc, argv);

  if (true)
  {
    printf("Running as a daemon\n");
    //Become a daemon
    // fork off the parent process
    pid_t pid = fork();
    if (pid < 0)
    {
      printf("Can not fork\n");
      exit(1);
    }

    if (pid > 0)
      exit(0); //exit the parent process

    // Change the file mask
    umask(0);

    //Create a new system id so that the kernel wont think we are an orphan.
    pid_t sid = setsid();
    if (sid < 0)
    {
      printf("Can not become independent\n");
      exit(1);
    }

    //fclose(stdin);
    //fclose(stdout);
    //fclose(stderr);

  }
  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  //start capturing
  open_device();
  init_device(0);
  start_capturing();






  CORBA::Object_var obj = orb->resolve_initial_references("RootPOA");
  PortableServer::POA_var poa = PortableServer::POA::_narrow(obj);

  SalGlassesServer* bcs = new SalGlassesServer(0); //debug value

  PortableServer::ObjectId_var bcsID = poa->activate_object(bcs);

  //get a ref string
  obj = bcs->_this();
  CORBA::String_var sior(orb->object_to_string(obj));
  std::cerr << "'" << (char*)sior << "'" << "\n";

  if( !bindObjectToName(orb, obj, OBJECT_NS, OBJECT_NAME, OBJECT_NAME, objectName) )
    return 1;

  bcs->_remove_ref();

  PortableServer::POAManager_var pman = poa->the_POAManager();
  pman->activate();

  orb->run();
  }

  catch(CORBA::SystemException& ex) {
    printf("Caught CORBA:: %s\n", ex._name());
  }
  catch(CORBA::Exception& ex) {
    printf("Caught CORBA::Exception: %s\n",ex._name());
  }
  catch(omniORB::fatalException& fe) {
    printf("Caught omniORB::fatalException:\n");
    printf("  file: \n", fe.file());
    printf("  line: \n", fe.line());
    printf("  mesg: \n", fe.errmsg());
  }

  printf("Shutting down\n");
  stop_capturing();
  uninit_device();
  close_device();
  unbindObject(orb, OBJECT_NS, OBJECT_NAME, objectName);

  return 0;

}

