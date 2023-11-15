/*!@file CINNIC/CINNICinteract.C Interactive CINNIC */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNICinteract.C $
// $Id: CINNICinteract.C 12074 2009-11-24 07:51:51Z itti $

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

//This is the start of the execution path for the CINNIC test alg.
#include "CINNIC/CINNIC.H"
#include "CINNIC/cascadeHold.H"
#include "Util/log.H"
#include "Util/readConfig.H"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <cstdio>




//! Create a CINNIC neuron, use float
ContourNeuronCreate<float> CINNICreate;
//! Standard CINNIC test object
CINNIC skeptic;
//! pointer to command line argument stream
std::istream* argInput;
//! This is the configFile name
const char* configFile;
//! This is the configFile object
readConfig configIn(25);
//! This is the main function for CINNIC
/*! to use this you must have a config file called contour.conf
 you should run this file by running it from src2 and executing it
 as such "../bin/CINNICtest ../image_in/imagefile outputname" where
 imagefile is a ppm file (do not include extention) and outputname
 is the name you would like attached to all the output files.
*/

int main(int argc, char* argv[])
{
  time_t t1,t2;
  (void) time(&t1);

  if(argc > 3)
  {
    configFile = argv[3];
  }
  else
  {
    configFile = "contour.conf";
  }
  skeptic.filename = "noName";
  LINFO("LOADING CONFIG FILE");
  configIn.openFile(configFile);
  for(int i = 0; configIn.readFileTrue(i); i++)
  {
    std::cout << configIn.readFileValueName(i) << ":" << configIn.readFileValueF(i) << "\n";
  }
  LINFO("Starting CINNIC interaction\n");
  CINNICreate.CreateNeuron(configIn);
  LINFO("CINNIC template Neuron copied, ready for use");

  if(argc > 2)
  {
    skeptic.savefilename = argv[2];
  }
  else
  {
    skeptic.savefilename = "noname";
  }

  LINFO("argc %d", argc);
  argInput = new std::istringstream(argv[1]);
  skeptic.Ninput = Image<byte>(100,100, NO_INIT);
  skeptic.Ninput = Raster::ReadGray(argv[1], RASFMT_PNM);
  skeptic.filename = argv[1];
  LINFO("Image Loaded");
  Image<float> temp;
  temp = skeptic.Ninput;
  LINFO("RUNNING TEST");
  skeptic.convolveTest(CINNICreate,configIn,temp);
  LINFO("done!");
  (void) time(&t2);
  long int tl = (long int) t2-t1;
  printf("\n*************************************\n");
  printf("Time to execute, %ld seconds\n",tl);
  printf("*************************************\n\n");
  return 1;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
