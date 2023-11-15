/*!@file CINNIC/CINNICtest.C  Tests the CINNIC neuron */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNICtest.C $
// $Id: CINNICtest.C 6410 2006-04-01 22:12:24Z rjpeters $
//

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
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

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
 you should run this file by running it from src3 and executing it
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
  /*for(int i = 0; configIn.readFileTrue(i); i++)
  {
    std::cout << configIn.readFileValueName(i) << ":" << configIn.readFileValueF(i) << "\n";
    }*/
  LINFO("Starting CINNIC test\n");
  CINNICreate.CreateNeuron(configIn);
  LINFO("CINNIC template Neuron copied, ready for use");
  int choice = 0;

  if(argc > 2)
  {
    skeptic.savefilename = argv[2];
  }
  else
  {
    skeptic.savefilename = "noname";
  }
  if(argc > 1) //command line argument
  {
    LINFO("argc %d", argc);
    argInput = new std::istringstream(argv[1]);
    skeptic.Ninput = Image<byte>(100,100, NO_INIT);
    skeptic.Ninput = Raster::ReadGray(argv[1], RASFMT_PNM);

    skeptic.filename = argv[1];
    LINFO("Image Loaded");
    skeptic.RunSimpleImage(CINNICreate,skeptic.Ninput,configIn);
    LINFO("done!");
  }
  else //no command line specified
  {
    std::ofstream outfile("kernel_set.dat",std::ios::out);
    choice = 0;
    while((choice < 9) && (choice > -1))
      {
      std::cout << "\n";
      std::cout << "iLab Global Dominanation Project (c) 2001\n";
      std::cout << "1 - Alpha Angle\n"
           << "2 - Beta Angle\n"
           << "3 - Distance\n"
           << "4 - Polarity\n"
           << "5 - Alpha/Beta Combo\n"
           << "6 - Alpha/Beta/Distance Combo\n"
           << "7 - coLinearity\n"
           << "8 - Visual ABD Combo\n"
           << "9 - QUIT\n\n"
           << "Choice: ";
      std::cin >> choice;
      if(choice == 8){skeptic.viewNeuronTemplate(CINNICreate,configIn);}
      if(choice < 8)
      {
        for (int i = 0; i < AnglesUsed; i++)
        {
          std::cout << "\n";
          outfile << "\n";
          for (int j = 0; j < AnglesUsed; j++)
          {
            std::cout << "\n\nNeuron:" << NeuralAngles[i] << " Other:" << NeuralAngles[j];
            for (int l = YSize; l >= 0; --l)
            {
              std::cout << "\n";
              outfile << "\n";
              for (int k = 0; k <= XSize; k++)
              {
                if(choice == 1)
                {
                  printf("%f ",CINNICreate.FourDNeuralMap[i][j][k][l].ang2);
                  outfile << CINNICreate.FourDNeuralMap[i][j][k][l].ang2 << "\t";
                }
                if(choice == 2)
                {
                  printf("%f ",CINNICreate.FourDNeuralMap[i][j][k][l].ang);
                  outfile << CINNICreate.FourDNeuralMap[i][j][k][l].ang << "\t";
                }
                if(choice == 3)
                {
                  printf("%f ",CINNICreate.FourDNeuralMap[i][j][k][l].dis);
                  outfile << CINNICreate.FourDNeuralMap[i][j][k][l].dis << "\t";
                }
                if(choice == 4)
                {
                  if(CINNICreate.FourDNeuralMap[i][j][k][l].pol)
                  {
                    std::cout << "1";
                  }
                  else
                  {
                    std::cout << "0";
                  }
                  std::cout << ":";
                  if(CINNICreate.FourDNeuralMap[i][j][k][l].sender)
                  {
                    std::cout << "1";
                  }
                  else
                  {
                    std::cout << "0";
                  }
                  std::cout << " ";
                }
                if(choice == 5)
                {
                  printf("%f ",CINNICreate.FourDNeuralMap[i][j][k][l].angAB);
                  outfile << CINNICreate.FourDNeuralMap[i][j][k][l].angAB << "\t";
                }
                if(choice == 6)
                {
                  printf("%f ",CINNICreate.FourDNeuralMap[i][j][k][l].angABD);
                  outfile << CINNICreate.FourDNeuralMap[i][j][k][l].angABD << "\t";
                }
                if(choice == 7)
                {
                  if(CINNICreate.FourDNeuralMap[i][j][k][l].coLinear)
                  {
                    std::cout << "1";
                  }
                  else
                  {
                    std::cout << "0";
                  }
                  std::cout << " ";
                }
              }
            }
          }
        }
      }
    }
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
