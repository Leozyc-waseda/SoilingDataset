/*!@file CINNIC/MrGabor.C A Gabor */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/MrGabor.C $
// $Id: MrGabor.C 9412 2008-03-10 23:10:15Z farhan $

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#include "CINNIC/gaborElement.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"   // for inplacePaste()
#include "Image/Image.H"
#include "Image/Kernels.H"    // for gaborFilter2()
#include "Image/Transforms.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Util/readConfig.H"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>


//! This is the configFile name
const char* configFile;
//! This is the gaborElement file
const char* gaborElementFile;
//! what to save the image as (name)
const char* imageSaveName;
//! This is the configFile object
readConfig configIn(25);
//! This is the gaborElement object
readConfig gaborIn(25);
//! keep track of gabor properties
std::vector<gaborElement> theGabors;
//! store the gabors
std::vector<Image<float> > theGaborImages;
//! the master images
std::vector<Image<float> > theMasterImages;
//! variables
int gaborNumber, sizeX, sizeY;
float backGround;
char getX[100];
char getY[100];
char getSTDDEV[100];
char getPERIOD[100];
char getPHASE[100];
char getTHETA[100];
char getSIGMOD[100];
char getAMPLITUDE[100];
char getMASTER[100];

int main(int argc, char* argv[])
{
  Image<float> finalImage;
  Image<float> masterImage;
  Image<byte> convert;
  time_t t1,t2;
  (void) time(&t1);
  configFile = "gabor.conf";
  gaborElementFile = argv[1];
  configIn.openFile(configFile);
  gaborIn.openFile(gaborElementFile);
  gaborNumber = (int)gaborIn.getItemValueF("gaborNumber");
  LINFO("Number of Gabor Elements in Image %d",gaborNumber);
  backGround = gaborIn.getItemValueF("backGround");
  LINFO("backGround color in Image %f",backGround);
  imageSaveName = gaborIn.getItemValueC("imageSaveName");
  LINFO("Will save as file %s",imageSaveName);
  sizeX = (int)gaborIn.getItemValueF("sizeX");
  sizeY = (int)gaborIn.getItemValueF("sizeY");
  LINFO("IMAGE size %dx%d",sizeX,sizeY);
  finalImage.resize(sizeX,sizeY);
  masterImage.resize(sizeX,sizeY);
  for(int x = 0; x < finalImage.getWidth(); x++)
  {
    for(int y = 0; y < finalImage.getHeight(); y++)
    {
      finalImage.setVal(x,y,backGround);
    }
  }
  gaborElement geTemp;
  theGabors.resize(gaborNumber);
  Image<float> giTemp;
  giTemp.resize(1,1);
  theGaborImages.resize(gaborNumber,giTemp);
  theMasterImages.resize(gaborNumber,giTemp);
  //go ahead and store gabors just incase we need them later
  LINFO("PLACING gabors...");
  for(int i = 0; i < gaborNumber; i++)
  {
    sprintf(getX,"x%d",i+1);
    theGabors[i].x = (int)gaborIn.getItemValueF(getX);
    sprintf(getY,"y%d",i+1);
    theGabors[i].y = (int)gaborIn.getItemValueF(getY);
    sprintf(getSTDDEV,"stddev%d",i+1);
    theGabors[i].stddev =
      gaborIn.getItemValueF(getSTDDEV);
    sprintf(getPERIOD,"period%d",i+1);
    theGabors[i].period =
      gaborIn.getItemValueF(getPERIOD);
    sprintf(getPHASE,"phase%d",i+1);
    theGabors[i].phase =
      gaborIn.getItemValueF(getPHASE);
    sprintf(getTHETA,"theta%d",i+1);
    theGabors[i].theta =
      gaborIn.getItemValueF(getTHETA);
    sprintf(getSIGMOD,"sigMod%d",i+1);
    theGabors[i].sigMod =
      gaborIn.getItemValueF(getSIGMOD);
    sprintf(getAMPLITUDE,"amplitude%d",i+1);
    theGabors[i].amplitude =
      gaborIn.getItemValueF(getAMPLITUDE);
    //sprintf(getMASTER,"master%d",i+1);
    //theGabors[i].master =
    //  (int)gaborIn.getItemValueF(getMASTER);
    theGabors[i].master = 0;
    theGaborImages[i] = gaborFilter2<float>(theGabors[i].stddev,
                                            theGabors[i].period,
                                            theGabors[i].phase,
                                            theGabors[i].theta,
                                            theGabors[i].sigMod,
                                            theGabors[i].amplitude);
    int xx = theGabors[i].x - (theGaborImages[i].getWidth()/2);
    int yy = theGabors[i].y - (theGaborImages[i].getHeight()/2);
    if(theGabors[i].master == 1)
    {
      theMasterImages[i] = gaussian2D<float>(theGabors[i].stddev,
                                             2.0F);
      //Raster::VisuFloat(theMasterImages[i], 0, "MASTER.pgm");
      inplacePaste(masterImage, theMasterImages[i],Point2D<int>(xx,yy));
    }
    inplacePasteGabor(finalImage, theGaborImages[i],
                      Point2D<int>(xx,yy),backGround);
  }
  //Raster::VisuFloat(finalImage, 0, imageSaveName, RASFMT_PNM);
  convert = finalImage;
  Raster::WriteGray(convert,sformat("%s.pgm", imageSaveName));
  //Raster::VisuFloat(masterImage, 0, sformat("%s.master.pgm",imageSaveName));
  convert = masterImage;
  Raster::WriteGray(convert,sformat("%s.master.pgm",imageSaveName));
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
