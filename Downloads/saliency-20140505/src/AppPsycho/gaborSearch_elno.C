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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/gaborSearch_elno.C $
// $Id: gaborSearch_elno.C 12962 2010-03-06 02:13:53Z irock $
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

typedef std::complex<float> complexf;

//! Return a noise added version of inputImg
Image<PixRGB<byte> > addNoise(const Image<PixRGB<byte> >& inputImg, int beta, float blendFactor, double *meanGray)
{

  //beta = 2 for random noise
  //beta = -2 pink(1/f) noise
  //beta = -4 brown noise

 bool owned;
 FILE* x =  stdOutputFileOpen("testnoise",&owned);
 FILE* y =  stdOutputFileOpen("testnoise_v",&owned);
 Image< PixRGB<byte> > theInputImg = inputImg;
 int inputW = theInputImg.getWidth(), inputH = theInputImg.getHeight(), squareDim;
 LINFO("input image inputW = %d h = %d",inputW,inputH);
  complexf il(0.0, 1.0);

  if(inputW > inputH)
    squareDim = inputW;
  else
    squareDim = inputH;

std::vector<float> u(squareDim*squareDim), v(squareDim*squareDim), spectrum(squareDim*squareDim),phi(squareDim*squareDim), resultVec(squareDim*squareDim);
  std::vector<complexf> sfRoot(squareDim*squareDim), cosPart(squareDim*squareDim), sinPart(squareDim*squareDim), cmplxResult(squareDim*squareDim);

  Dims slightlyBigger ((squareDim*2)-1, squareDim);

  FourierInvEngine<double> ieng(slightlyBigger);

  int crnt = 0, v_cnt = 0;
  //create u and v vectors (matrix abstractions) where v = transpose of u

  for(int i=0; i < squareDim/2; i++)
     {
       u[i] = i+1;
       crnt  = crnt + 1;
     }


  for(int j = - squareDim/2; j < 0; j++)
      u[crnt++] = j;

  for(int i=1; i<squareDim; i++)
    for(int j=0; j<squareDim; j++)
      {
        u[crnt++] = u[j];
        v[v_cnt]  = u[i-1];
        v_cnt     = v_cnt + 1;
      }


  for(int j=0; j<squareDim ;j++)
    {
      v[v_cnt] = u[squareDim-1];
      v_cnt = v_cnt + 1;
    }

  Image<std::complex<float> > tempCmplxImg(squareDim,squareDim,NO_INIT);
  Image <complexf>::iterator cmplxItr =tempCmplxImg.beginw();



  complexf tempcf;
  for(int i=0; i< (squareDim*squareDim); i++)
   {
      spectrum[i] = pow((u[i]*u[i]) + (v[i]*v[i]),(beta/2));
      srand(time(NULL) + 10*i);
      phi[i] = ((double)rand()/((double)(RAND_MAX)+(double)(1)) );
      sfRoot[i]      = complexf(pow(spectrum[i],0.5),0.0);
      cosPart[i]     = complexf(cos(2*M_PI*phi[i]),  0.0);
      sinPart[i]     = complexf(sin(2*M_PI*phi[i]),  0.0);
      tempcf =  (cosPart[i] + il*sinPart[i]);
      cmplxResult[i] = sfRoot[i]*tempcf ;
     *cmplxItr++    = cmplxResult[i];

   }


  crnt =0;
  for(int i=0; i<squareDim; i++)
    {   fprintf(x,"\n");
      for(int j=0; j<squareDim; j++)
        {
          fprintf(x,"\tu[%d]=%f",crnt,u[crnt]);
          crnt = crnt +1;
        }
    }


  Image<double> res = ieng.ifft(tempCmplxImg);
  Image <double>::iterator apItr = res.beginw();

  double imgMax,imgMin;
  getMinMax(res,imgMin,imgMax);

  apItr = res.beginw();

  for(int i=0; i<slightlyBigger.w()*slightlyBigger.h(); i++)
   {
     *apItr =  ((*apItr - imgMin)/(imgMax - imgMin))*255;
      apItr++;
   }

 apItr = res.beginw();
 crnt = 0;
 for(int i=0; i<squareDim; i++)
    {   fprintf(y,"\n");
      for(int j=0; j<squareDim; j++)
        {
          fprintf(y,"\tv[%d]=%f",crnt,v[crnt]);
          crnt = crnt +1;
        }
    }

 *meanGray = mean(res);
 Image<PixRGB<byte> > resPixRGB = res;

resPixRGB = rescale(resPixRGB,Dims (inputW,inputH));

 LINFO("input image w = %d h = %d",resPixRGB.getWidth(),resPixRGB.getHeight());

blendFactor=0;
Image<PixRGB<byte> > resByte = theInputImg + resPixRGB*blendFactor;
// Image<PixRGB<byte> > resByte = theInputImg + resPixRGB*blendFactor;


  fclose(x);
  fclose(y);
  return resByte;

}


int main(int argc, char** argv)
{


  clock_t t1=  clock();
  //instantiate a model manager
  ModelManager manager("AppPsycho: gabor search stimuli");

  //dimensions of window
  //Dims dims(1920,1080);
Dims dims(1920,1080);

  float stddev  = 150.0;
  float freq    = 5.0;
  float theta1  = 10;
  float hueShift= 5.0;
  float f,t,h;
  int gaborSize   = 240;
  int n_gabors_w  = dims.w()/gaborSize;
  int n_gabors_h  = dims.h()/gaborSize;
  int totalGabors = n_gabors_w * n_gabors_h;
  char filename[255],fname[255];

  Image<PixRGB<byte> > dispImg(1920,1080,NO_INIT);

  if (manager.parseCommandLine(argc, argv,"<name>", 1, 1) == false)
    return(1);

  manager.start();

  // get command line parameters
  sscanf(argv[1],"%s",fname);

  sprintf(filename, "/lab/elno/research/exp1/spec/%sSPEC.dat",fname);
  FILE* specFile = fopen(filename, "w+");
  if(specFile==0)
    LFATAL("couldnt open file:%s", filename);

  fprintf(specFile,"stddev = %g; freq = %g; theta = %g; hueShift = %g",stddev,freq,theta1,hueShift);

  //make a 2d vector of images//

  std::vector< Image<PixRGB<byte> > > ivector(totalGabors + 1);
  std::vector< Image<PixRGB<byte> > > jvector(totalGabors + 1);
  std::vector< Image<PixRGB<byte> > > constructArray(totalGabors + 1);
  std::vector<Point2D<int> > gaborItems(totalGabors + 1);  //vector to hold xy coordinates of the gabor Patches
  std::vector<Point2D<int> > circleItems(totalGabors + 1);  //vector to hold xy coordinates of the circle
  std::vector<Point2D<int> > randPos(totalGabors + 1);  //vector to hold xy coordinates of the gabor Patches
  std::vector<int> used (totalGabors + 1);
  Dims standarddims(gaborSize,gaborSize);

  srand(time(NULL));
  //lets create some random positions
  int cnt =1;
  int yOffset =50; //yOffset required to compensate for the gaborSize/2 freespace
 for(int j = 1; j <= n_gabors_h; j++)
    for (int i = 1; i <= n_gabors_w; i++)
      {
         if(i>1 && i<n_gabors_w-1 && j>1 && j<n_gabors_h-1)
           randPos[cnt] =   Point2D<int>((i*gaborSize-gaborSize) +
                                    (rand()*10)%(75),(j*gaborSize-gaborSize) + (rand()*10)%(75) + yOffset);
         else
           randPos[cnt] = Point2D<int>(i*gaborSize-gaborSize,j*gaborSize-gaborSize + yOffset);
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
     //drawDisk(constructArray[cnt], gaborItems[cnt]+gaborSize/2 ,gaborSize/4 , PixRGB<byte>(255,255,  0));
 drawDisk(ivector[cnt], Point2D<int> (0,0)+gaborSize/2 ,gaborSize/3 , PixRGB<byte>(128,128,128));

        ivector[cnt] = rescale(ivector[cnt],standarddims);
 ivector[cnt] = ivector[cnt]+ rescale(gaborFilterRGB(stddev, f,t,h),standarddims);

      //  jvector[cnt] = gaborFilterRGB(stddev, f,t,h);
       // jvector[cnt] = rescale(ivector[cnt],standarddims);




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
         constructArray[cnt] = Image<PixRGB<byte> >(1920,1080,NO_INIT);
//drawDisk(constructArray[cnt], gaborItems[cnt]+gaborSize/2 ,gaborSize/4 , PixRGB<byte>(255,255,  0));
         inplacePaste(constructArray[cnt],ivector[cnt],gaborItems[cnt]);
         cnt++;
      }

  //lets put them together
  for (int i=1; i < cnt; i++)
    dispImg = dispImg + constructArray[i]*1;




  //lets add some noise to the image and save it
  Image <PixRGB<byte> > myResult;

 cnt=1;
  for(int j = 1; j <= n_gabors_h; j++)
    for (int i = 1; i <= n_gabors_w; i++)
      {
//drawDisk(dispImg, gaborItems[cnt],gaborSize/2 , PixRGB<byte>(255,255,  0));
      }


  double meanGray=0.0;
  myResult = addNoise(dispImg,-2,0.75, &meanGray);
  fprintf(specFile,"\nGray %f",meanGray);
  sprintf(filename, "/lab/elno/research/exp1/stim/%sARRAY.png",fname);

cnt=1;
  for(int j = 1; j <= n_gabors_h; j++)
    for (int i = 1; i <= n_gabors_w; i++)
      {
//drawDisk(myResult, gaborItems[cnt],gaborSize/2 , PixRGB<byte>(255,255,  0));
      }
Raster::WriteRGB(myResult,filename);

  //create save the target image

  Image<double> target(1920,1080,ZEROS);
  Image<PixRGB<byte> > targetBkg(1920,1080,ZEROS);

  srand(time(NULL));
  randIndex = rand() % totalGabors + 1;
  fprintf(specFile,"\n\ntarget %d",randIndex);

  inplacePaste(targetBkg, ivector[randIndex],Point2D<int>((1920/2)-(gaborSize/2),(1080/2)-(gaborSize/2)));
  Image<double>::iterator aptr= target.beginw();

  while(aptr!=target.endw())
    *aptr++ = meanGray;

  Image<PixRGB<byte> > targetRGB = target;
  targetRGB = targetRGB + targetBkg;


//   cnt=1;
//   for(int j = 1; j <= n_gabors_h; j++)
//     for (int i = 1; i <= n_gabors_w; i++)
//       {drawDisk(targetRGB, gaborItems[cnt],gaborSize/2 , PixRGB<byte>(0,0,0));
//       }


  char targetFileName[255];
  sprintf(targetFileName, "/lab/elno/research/exp1/stim/%sTARGET.png",fname);
  Raster::WriteRGB(targetRGB,targetFileName);

  //create the screen for reporting position of target gabor

  Image<PixRGB<byte> > reportDot(1920,1080,ZEROS);
  reportDot = target;

  for(int i=1;i<=totalGabors;i++)
    {
      //  drawDisk(reportDot, gaborItems[i]+(gaborSize/2), 10,PixRGB<byte>(255,255,255));
      writeText(reportDot, gaborItems[i]+(gaborSize/2)-10, sformat("%d",i).c_str(),
                PixRGB<byte>(0,0,0),PixRGB<byte>(0),SimpleFont::FIXED(12) ,true);
    }

  char reportDotFile[255];
  sprintf(reportDotFile, "/lab/elno/research/exp1/stim/%sREPORT.png",fname);
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
