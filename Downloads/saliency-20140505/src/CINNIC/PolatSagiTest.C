/*!@file CINNIC/PolatSagiTest.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/PolatSagiTest.C $
// $Id: PolatSagiTest.C 6410 2006-04-01 22:12:24Z rjpeters $

#include "Util/Assert.H"
#include "CINNIC/CINNICstatsRun.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"

#include <fstream>
#include <iostream>

//! this contains the name of the config file
const char* configFile;
//! this contains the name of the second config file
const char* configFile2;
//! this is the target file to open
const char* targetFileName;
//! this is the non target file to open
const char* notargetFileName;
//! This is the configFile object
readConfig ReadConfig(25);
//! This is the second configFile object
readConfig ReadConfig2(25);
//! main object for stats
CINNICstatsRun runStats;
//! image float
Image<float> Fimage[2];
//! float for return
float forReturn;
//! input string for parsing input files
std::string in;
//! image input data
int height, width;
//! image input data
int currentX, currentY;
//! This is a package to run tests on the output from CINNIC
int main(int argc, char* argv[])
{
  LINFO("Polat and Sagi 2AFC test");
  LINFO("Copyright 2002 ACME Vision Systems, Wallawalla, WA");
  ASSERT(argc > 1);
  targetFileName = argv[1];
  notargetFileName = argv[2];

  if(argc > 3)
  {
    configFile = argv[3];
  }
  else
  {
    configFile = "stats.conf";
  }
  if(argc > 4)
  {
    configFile2 = argv[4];
  }
  else
  {
    configFile2 = "contour.conf";
  }

  ReadConfig.openFile(configFile);
  ReadConfig2.openFile(configFile2);
  LINFO("Setting Stuff in Stats");
  runStats.setStuff();
  LINFO("Setting Stuff in Config");
  runStats.setConfig(ReadConfig,ReadConfig2);

  std::ifstream readData;
  int item;

  for(int i = 0; i < 2; i++)
  {
    if(i == 0)
    {
      readData.open(targetFileName,std::ios::in);
      LINFO("Opening %s",targetFileName);
    }
    if(i == 1)
    {
      readData.open(notargetFileName,std::ios::in);
      LINFO("Opening %s",notargetFileName);
    }
    //parse in and read data from file into an image
    item = 0;
    while (readData >> in)
    {
      if(item == 0)
      {
        width = atoi(in.c_str());
        currentX = 0;
      }
      if(item == 1)
      {
        height = atoi(in.c_str());
        currentY = 0;
      }
      if(item == 2)
        Fimage[i].resize(width,height);
      if(item >= 2)
      {
        if(currentY == height)\
        {
          currentY = 0;
          currentX++;
        }
        Fimage[i].setVal(currentX,currentY,atof(in.c_str()));
        currentY++;

      }
      item++;
    }
    readData.close();
    readData.clear();
  }


  LINFO("DUH, they all look the same");
  forReturn = runStats.polatSagi2AFC(Fimage[0],Fimage[1]);
  LINFO("Probability Value = %f",forReturn);
  std::cout << forReturn << "\n";
  std::cout << runStats.getMu1() << "\n";
  std::cout << runStats.getMu2() << "\n";
  return (int)(forReturn*10000);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
