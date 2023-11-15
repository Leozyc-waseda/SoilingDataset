/*!@file CINNIC/runCINNIC2.C  this runs CINNIC2 */

// //////////////////////////////////////////////////////////////////// //
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
// Primary maintainer for this file: T Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/runCINNIC2.C $
// $Id: runCINNIC2.C 12074 2009-11-24 07:51:51Z itti $
//

// ############################################################
// ############################################################
// ##### ---CINNIC2---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

//This is the start of the execution path for the CINNIC test alg.
#include "Util/log.H"
#include "Util/readConfig.H"

#include "CINNIC/CINNIC2.H"

#include <stdlib.h>
#include <string>
#include <sys/types.h>
#include <time.h>
#include <cstdio>

//! Create a CINNIC neuron, use float
ContourNeuronCreate<float> CINNICreate;
//! pointer to command line argument stream
std::istream* argInput;
//! This is the configFile name
const char* configFile;
//! This is the configFile object
readConfig configIn(25);

//! This is the main binary for CINNIC2
int main(int argc, char* argv[])
{
  LINFO("RUNNING CINNIC 2");
  time_t t1,t2;
  (void) time(&t1);

  bool         useFrames    ;
  unsigned int frames    = 0;
  if(argc > 4)
  {
    useFrames = true;
    frames    = atoi(argv[4]);
  }
  else
  {
    useFrames = false;
  }

  if(argc > 3)
  {
    configFile = argv[3];
  }
  else
  {
    configFile = "contour.conf";
  }
  LINFO("LOADING CONFIG FILE");
  configIn.openFile(configFile);

  LINFO("Starting CINNIC test\n");
  CINNICreate.CreateNeuron(configIn);
  LINFO("CINNIC template Neuron copied, ready for use");

  std::string savefilename;

  if(argc > 2)
  {
    savefilename = argv[2];
  }
  else
  {
    savefilename = "noname";
  }
  if(argc > 1) //command line argument
  {
    unsigned int currentFrame = 1;
    CINNIC2<XSize,Scales,AnglesUsed,Iterations,float,int> skeptic;
    LINFO("argc %d", argc);
    Image<byte> inputImage;
    if(useFrames == false)
    {

      inputImage = Raster::ReadGray(argv[1], RASFMT_PNM);
      std::string filename = argv[1];
      LINFO("Image Loaded");
      skeptic.CINrunSimpleImage(CINNICreate,filename.c_str(),currentFrame
                                ,inputImage,configIn);
      LINFO("done!");
    }
    else
    {
      Image<PixRGB<byte> > inputImageC;
      for(unsigned int i = 0; i < frames; i++)
      {
        skeptic.CINtoggleFrameSeries(true);
        std::string filename;
        std::string a = argv[1];
        char c[100];
        if(i < 10)
          sprintf(c,"00000%d",i);
        else if(i < 100)
          sprintf(c,"0000%d",i);
        else if(i < 1000)
          sprintf(c,"000%d",i);
        else if(i < 10000)
          sprintf(c,"00%d",i);
        else if(i < 100000)
          sprintf(c,"0%d",i);
        else
          sprintf(c,"%d",i);
        filename    = a + c;
        inputImageC = Raster::ReadRGB(filename, RASFMT_PNM);
        inputImage  = luminance(inputImageC);
        LINFO("Image %s Loaded",filename.c_str());
        skeptic.CINrunSimpleImage(CINNICreate,filename.c_str(),currentFrame
                                ,inputImage,configIn);
        currentFrame++;
      }
      LINFO("done!");
    }
  }
  else
  {
    LINFO("You must specify command line args");
    LINFO("runCINNIC in_file out_file config_file");
  }

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
