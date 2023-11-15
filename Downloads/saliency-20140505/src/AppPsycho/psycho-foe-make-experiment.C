/*!@file AppPsycho/gaborSearch.C                       */
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
// Primary maintainer for this file: Elnaz Nouri <enouri@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-foe-make-experiment.C $
// $Id: psycho-foe-make-experiment.C 12962 2010-03-06 02:13:53Z irock $
//

//generate random search arrays of colored gabor patches using the gaborfilterRGB function

#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/FourierEngine.H"
#include "Image/Image.H"
#include "Image/Kernels.H"     // for gaborFilter()
#include "Image/MathOps.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include "Image/Transforms.H"
#include "Raster/PngWriter.H"
#include "Raster/Raster.H"
#include "Util/FileUtil.H"
#include "Util/log.H"
#include <stdio.h>
#include <algorithm>
#include <time.h>


#define WIDTH 1280
#define HEIGHT 720

typedef std::complex<float> complexf;

int main(int argc, char** argv)
{
  clock_t t1=  clock();
  //instantiate a model manager
  ModelManager manager("AppPsycho: gabor search stimuli");

  //dimensions of window
  Dims dims(WIDTH,HEIGHT);

  float stddev  = 150.0;
  float freq    = 5.0;
  float theta1  = 10;
  float hueShift= 5.0;
  float f,t,h;
  int gaborSize   = 320;
  int n_gabors_w  = dims.w()/gaborSize;
  int n_gabors_h  = dims.h()/gaborSize;
  int totalGabors = n_gabors_w * n_gabors_h;

  char filename[255],setnumber[255] ,imagenumber[255];

  Image<PixRGB<byte> > dispImg(WIDTH,HEIGHT,NO_INIT);

 if (manager.parseCommandLine(argc, argv,"<setnumber> <imagenumber>", 2, 2) == false){

   return(1);
}
  manager.start();

  // get command line parameters
  sscanf(argv[1],"%s",setnumber);
  sscanf(argv[2],"%s",imagenumber);
  sprintf(filename, "/home2/tmp/u/elno/research/exp1/spec/set%s/%sSPEC.dat",setnumber, imagenumber);
  FILE* specFile = fopen(filename, "w+");
  if(specFile==0)
    LFATAL("couldnt open file:%s", filename);
  fprintf(specFile,"stddev = %g; freq = %g; theta = %g; hueShift = %g",stddev,freq,theta1,hueShift);

  //make a 2d vector of images//

  std::vector< Image<PixRGB<byte> > > ivector(totalGabors + 1); // vector of gabors
  std::vector< Image<PixRGB<byte> > > constructArray(totalGabors + 1);
  std::vector<Point2D<int> > gaborItems(totalGabors + 1);  //vector to hold xy coordinates of the gabor Patches
  std::vector<Point2D<int> > circleItems(totalGabors + 1);  //vector to hold xy coordinates of the circle
  std::vector<Point2D<int> > randPos(totalGabors + 1);  //vector to hold xy coordinates of the gabor Patches
  std::vector<int> used (totalGabors + 1);
  Dims standarddims(gaborSize,gaborSize);

  srand(time(NULL));
  //lets create some random positions
  int cnt =1;
  int xOffset =0;
  int yOffset =40; //yOffset required to compensate for the gaborSize/2 freespace
 for(int j = 1; j <= n_gabors_h; j++)
    for (int i = 1; i <= n_gabors_w; i++)
      {
         if(i>1 && i<n_gabors_w-1 && j>1 && j<n_gabors_h-1)
           randPos[cnt] =   Point2D<int>((i*gaborSize-gaborSize)+(rand()*15-5) +xOffset ,(j*gaborSize-gaborSize)+(rand()*15-5) + yOffset);
         else
           randPos[cnt] = Point2D<int>(i*gaborSize-gaborSize + xOffset ,j*gaborSize-gaborSize + yOffset);
         cnt++;
      }

  //create gabor patches and assign pseudorandom xy positions.
  std::vector<int>::iterator result;
  srand(time(NULL));
  int randIndex = rand() % totalGabors + 1;
  used.push_back(randIndex);
  result = used.end();

  cnt=1;
  for(int j = 1; j <= n_gabors_h; j++)
    for (int i = 1; i <= n_gabors_w; i++)
      {

        //frequencies can vary between 0.0075 and 0.0225 (0.0065-0.0075, and 0.0225-0.0250 reserved for post-training testing)
        f = ((rand()*(float(225-75)/RAND_MAX)) + 75)/10000;
        //orientation can vary between 15-75 deg (0-15 deg and 75-90 reserved for post-training testing)
        t = (rand()*(float(155-25)/RAND_MAX)) + 25;
        //hue shifts can vary between 0-360
        h = (rand()*(((float)360/RAND_MAX)));

        ivector[cnt] = Image<PixRGB<byte> >(gaborSize,gaborSize,NO_INIT);
        ivector[cnt].clear(PixRGB<byte>(128,128,128));
        //drawDisk(constructArray[cnt], gaborItems[cnt]+gaborSize/2 ,gaborSize/4 , PixRGB<byte>(255,255,  0));
        //drawDisk(ivector[cnt], Point2D<int> (gaborSize/2,gaborSize/2) ,gaborSize/4 , PixRGB<byte>(128,128,128));
        ivector[cnt] = rescale(ivector[cnt],standarddims);
        //sprintf(filename, "/home2/tmp/u/elno/research/exp1/stim/fgImages/set%s/%szgab_%d_%d.png",setnumber, imagenumber, i,j);
        //Raster::WriteRGB(ivector[cnt],filename);

        ivector[cnt] = ivector[cnt]+ rescale(gaborFilterRGB(stddev, f,t,h),standarddims);

        //sprintf(filename, "/home2/tmp/u/elno/research/exp1/stim/fgImages/set%s/%sz2gab_%d_%d.png",setnumber, imagenumber, i,j);
        //Raster::WriteRGB(ivector[cnt],filename);

        //MASK IT
        for (int a = 0; a <= gaborSize; ++a)
        for (int b = 0; b <= gaborSize; ++b)
        {
                if( (squareOf(a-gaborSize/2) + squareOf(b-gaborSize/2)) > squareOf(gaborSize/4) )
                        if (ivector[cnt].coordsOk(a, b))
                                ivector[cnt].setVal(a, b, PixRGB<byte>(0,0,0));
        }

        while(result!=used.end())
          {
            srand(time(NULL));
            randIndex = rand() % totalGabors + 1;
            result = find(used.begin(),used.end(),randIndex);
          }

        used.push_back(randIndex);
        gaborItems[cnt] = randPos[randIndex];
        fprintf(specFile,"\n\nGabor %d",cnt);
        fprintf(specFile,"\nXpos\t%d\nYpos\t%d\nstddev\t%.0f\nHueShift\t%.0f\nfrequency\t%f\nOrientation\t%.0f",
        gaborItems[cnt].i,gaborItems[cnt].j,stddev,h,f*1000,t);
        cnt++;
      }

  //lets paste the gabors into one big image
  cnt=1;
  for(int j = 1; j <= n_gabors_h; j++)
    for (int i = 1; i <= n_gabors_w; i++)
      {
        constructArray[cnt] = Image<PixRGB<byte> >(WIDTH,HEIGHT,NO_INIT);
        constructArray[cnt].clear(PixRGB<byte>(0,0,0));
         inplacePaste(constructArray[cnt],ivector[cnt],gaborItems[cnt]);
          //sprintf(filename, "/lab/ilab19/elnaz/research/exp1/stim/fgImages/set%s/%szconst_%d_%d.png",setnumber, imagenumber, i,j);
        //Raster::WriteRGB(constructArray[cnt],filename);
         cnt++;
      }

dispImg.clear(PixRGB<byte>(0,0,0));
  //lets put them together
  for (int i=1; i < cnt; i++){
    dispImg = dispImg + constructArray[i]*1;
        //sprintf(filename, "/lab/ilab19/elnaz/research/exp1/stim/fgImages/set%s/%szzzzzz_%d.png",setnumber, imagenumber, i);
        //Raster::WriteRGB(dispImg,filename);
}


  Image <PixRGB<byte> > myResult;
  double meanGray=0.0;
  myResult = dispImg;
  fprintf(specFile,"\nGray %f",meanGray);
  sprintf(filename, "/home2/tmp/u/elno/research/exp1/stim/fgImages/set%s/%sARRAY.png",setnumber, imagenumber);
  Raster::WriteRGB(myResult,filename);

  //create save the target image
  Image<double> target(WIDTH,HEIGHT,ZEROS);
  Image<PixRGB<byte> > targetBkg(WIDTH,HEIGHT,ZEROS);

  srand(time(NULL));
  randIndex = rand() % totalGabors + 1;
  fprintf(specFile,"\n\ntarget %d",randIndex);

  inplacePaste(targetBkg, ivector[randIndex],Point2D<int>((WIDTH/2)-(gaborSize/2),(HEIGHT/2)-(gaborSize/2)));
  Image<double>::iterator aptr= target.beginw();

  Image<PixRGB<byte> > targetRGB = target;
  targetRGB = targetRGB + targetBkg;

//MASK IT
  for (int a = 0; a <= WIDTH; ++a)
  for (int b = 0; b <= HEIGHT; ++b)
    {
        if( (squareOf(a-WIDTH/2) + squareOf(b-HEIGHT/2)) > squareOf(gaborSize/4) )
                if (targetRGB.coordsOk(a, b))
                          targetRGB.setVal(a, b, PixRGB<byte>(128,128,128));
    }

  while(aptr!=target.endw())
    *aptr++ = meanGray;


  char targetFileName[255];
  sprintf(targetFileName, "/home2/tmp/u/elno/research/exp1/stim/fgImages/set%s/%sTARGET.png",setnumber, imagenumber);
  Raster::WriteRGB(targetRGB,targetFileName);

  //create the screen for reporting position of target gabor
  Image<PixRGB<byte> > reportDot(WIDTH,HEIGHT,ZEROS);
  reportDot.clear(PixRGB<byte>(128,128,128));
  for(int i=1;i<=totalGabors;i++)
    {
      //  drawDisk(reportDot, gaborItems[i]+(gaborSize/2), 10,PixRGB<byte>(255,255,255));
      writeText(reportDot, gaborItems[i]+(gaborSize/2)-10, sformat("%d",i).c_str(),
                PixRGB<byte>(0,0,0),PixRGB<byte>(0),SimpleFont::FIXED(12) ,true);
    }

  char reportDotFile[255];
  sprintf(reportDotFile, "/home2/tmp/u/elno/research/exp1/stim/fgImages/set%s/%sREPORT.png",setnumber, imagenumber
);
  Raster::WriteRGB(reportDot, reportDotFile);

  clock_t t2=  clock();
  LINFO("generated search array in %fs", double(t2-t1)/CLOCKS_PER_SEC);

  fclose(specFile);
  //finish all
  manager.stop();
  return 0;

}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
