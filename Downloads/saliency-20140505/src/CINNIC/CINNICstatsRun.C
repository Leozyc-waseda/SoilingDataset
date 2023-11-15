/*!@file CINNIC/CINNICstatsRun.C run anal. of CINNIC */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNICstatsRun.C $
// $Id: CINNICstatsRun.C 14376 2011-01-11 02:44:34Z pez $
//

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################
#include "CINNIC/CINNICstatsRun.H"

#include "Util/Assert.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Transforms.H"

#include <fstream>

CINNICstatsRun::CINNICstatsRun()
{
}

CINNICstatsRun::~CINNICstatsRun()
{
}



void CINNICstatsRun::setConfig(readConfig &config, readConfig &config2)
{
  /* FORMAT...
     iterations = (int)config.getItemValueF("iterations");
     ...
  */
  LINFO("READING CONFIG VALUES");
  decSize = (int)config.getItemValueF("decSize");
  unEvenDec = (int)config.getItemValueF("unEvenDec");
  lowThresh = (int)config.getItemValueF("lowThresh");
  highThresh = (int)config.getItemValueF("highThresh");
  chopVal = (int)config.getItemValueF("chopVal");
  pointsNum = (int)config.getItemValueF("pointsNum");
  circRad = (int)config.getItemValueF("circRad");
  floodThresh = config.getItemValueF("floodThresh");
  floodVal = config.getItemValueF("floodVal");
  qualityImage = (int)config.getItemValueF("qualityImage");
  qChop = (int)config.getItemValueF("qChop");
  doResize = (int)config.getItemValueF("doResize");
  rankThresh = (int)config.getItemValueF("rankThresh");
  preProcessPNF = (int)config.getItemValueF("preProcessPNF");
  useDrawDisk = (int)config.getItemValueF("useDrawDisk");
  diskSize = (int)config.getItemValueF("diskSize");
  statsFile = config.getItemValueS("statsFile");
  monoColor = (int)config.getItemValueF("monoColor");
  PSError = config2.getItemValueF("PSError");
  centerOffset = (int)config.getItemValueF("centerOffset");
  edgeAtten = (int)config.getItemValueF("edgeAtten");
}

void CINNICstatsRun::setStuff(readConfig &fileList)
{
  tempImage1c.resize(1,1);
  tempImage2.resize(1,1);
  salMap.resize(1,1);
  point2D = new Point2D<int>;
  pointVec.resize(5,*point2D);
  maxPointList.resize(fileList.itemCount(),pointVec);
  postChamferVal.resize(fileList.itemCount(),0.0F);
  errorCount.resize(fileList.itemCount(),0);
  totCount.resize(fileList.itemCount(),0);
  totalVal.resize(fileList.itemCount(),0.0F);
  errorRatio.resize(fileList.itemCount(),0.0F);
  testMean.resize(fileList.itemCount(),0.0F);
  testStd.resize(fileList.itemCount(),0.0F);
  compMean.resize(fileList.itemCount(),0.0F);
  compStd.resize(fileList.itemCount(),0.0F);
  regression.resize(fileList.itemCount(),0.0F);
  eucDist.resize(fileList.itemCount(),0.0F);
  pointsFound.resize(fileList.itemCount(),0);
  pointsFoundNum.resize(pointsNum,0);
  pointRank.resize(fileList.itemCount(),-1);
  pointRankNum.resize(pointsNum,0);
  candPixels.resize(pointsNum,0);
  realPixels.resize(pointsNum,0);
  strikeP.resize(pointsNum,0.0F);
  bernoulliP.resize(pointsNum,0.0F);
  totalErrorRatio = 0;  maxMean = 0; minMean = 1;
  totEucDist = 0; maxEuc = 0 ; minEuc = 1;
  totReg = 0; maxReg = 0; minReg = 1;
  zeroCount = 0;
  foundTotal = 0.0F; rankTotal = 0.0F;
  foundMean = 0.0F; rankMean = 0.0F;
}

void CINNICstatsRun::setStuff()
{
  tempImage1c.resize(1,1);
  tempImage2.resize(1,1);
  salMap.resize(1,1);
  point2D = new Point2D<int>;
  pointVec.resize(5,*point2D);
  pointRankNum.resize(pointsNum,0);
  totalErrorRatio = 0;  maxMean = 0; minMean = 1;
  totEucDist = 0; maxEuc = 0 ; minEuc = 1;
  totReg = 0; maxReg = 0; minReg = 1;
  zeroCount = 0;
  foundTotal = 0.0F; rankTotal = 0.0F;
  foundMean = 0.0F; rankMean = 0.0F;
  pointsFoundNum.resize(pointsNum,0);
}

void CINNICstatsRun::runStandardStats(readConfig &fileList)
{
  //! point2D object
  //Point2D<int> *point2D = new Point2D<int>();
  imageCount = fileList.itemCount();
  const char* inCompImage;

  //iterate over each image pair
  for(int i = 0; fileList.readFileTrue(i); i++)
  {
    // pair must exist
    LINFO("PAIR %d",i);
    // get names of images from file list using readConfig
    fileList.readFileValueNameC(i);
    inCompImage = fileList.readFileValueC(i);
    // read the images
    tempImage1c = Raster::ReadRGB(fileList.readFileValueNameC(i), RASFMT_PNM);
    LINFO("DONE");
    tempImage2 = Raster::ReadGray(fileList.readFileValueC(i), RASFMT_PNM);
    LINFO("DONE");
    // Make sure they are good
    checkSize();

    // format them
    tempImage1cf = tempImage1c;
    testImage = luminance(tempImage1cf);
    inplaceAttenuateBorders(testImage, edgeAtten);
    compImage = tempImage2;
    //Raster::VisuFloat(testImage, FLOAT_NORM_0_255, sformat("foo.%d.pgm",i));
    //Raster::VisuFloat(compImage, FLOAT_NORM_0_255, sformat("bar.%d.pgm",i));
    preProcess();

    //Calculate basic stats
    testImageVector.assign(testImage.begin(), testImage.end());
    compImageVector.assign(compImage.begin(), compImage.end());
    testMean[i] = Stats.mean(testImageVector);
    testStd[i] = Stats.findS(testImageVector,testMean[i]);
    LINFO("testMean is %f std %f",testMean[i],testStd[i]);
    compMean[i] = Stats.mean(compImageVector);
    compStd[i] = Stats.findS(compImageVector,compMean[i]);
    LINFO("compMean is %f std %f",compMean[i],compStd[i]);
    regression[i] = Stats.rRegression(testImageVector,compImageVector);


    // Find the euclidian distance between images
    for(int x = 0; x < testImage.getWidth(); x++)
    {
      for(int y = 0; y < testImage.getHeight(); y++)
      {
        eucDist[i] += pow((testImage.getVal(x,y) - compImage.getVal(x,y)),2);
      }
    }

    //choped for ease of use
    float hold  = (1.0F/(testImage.getWidth()*testImage.getHeight()));
    eucDist[i] = hold*sqrt(eucDist[i]);

    // polarize pixels to either 0 or 255 based on mean
    for(int x = 0; x < testImage.getWidth(); x++)
    {
      for(int y = 0; y < testImage.getHeight(); y++)
      {
        if(testImage.getVal(x,y) < testMean[i])
        {
          testImage.setVal(x,y,0.0F);
        }
        else
        {
          testImage.setVal(x,y,254.0F);
        }
      }
    }

    //Raster::VisuFloat(compImage, FLOAT_NORM_0_255, sformat("bar3.%d.pgm",i));

    //LEVITY I DEMAND LEVITY

    //Raster::VisuFloat(testImage, FLOAT_NORM_0_255, sformat("foo.%d.pgm",i));
    //Raster::VisuFloat(compImage, FLOAT_NORM_0_255, sformat("bar.%d.pgm",i));
    //compImage = rescale(compImage,decSize,decSize);
    //testImage = rescale(testImage,decSize,decSize);
    //totCount[i] = 0;
    //errorCount[i] = 0;

    // count for error pixels
    for(int x = 0; x < compImage.getWidth(); x++)
    {
      for(int y = 0; y < compImage.getHeight(); y++)
      {
        if(testImage.getVal(x,y) > 1)
        {
          totCount[i]++;
          if(compImage.getVal(x,y) < 1)
          {
            errorCount[i]++;
          }
        }
      }
    }
    //float avg = totalVal[i]/(compImage.getWidth()*compImage.getHeight());
    if((errorCount[i] != 0) && (totCount[i] != 0))
    {
      errorRatio[i] = (float)errorCount[i]/(float)totCount[i];
    }
    else
    {
      errorRatio[i] = 0.0F;
      zeroCount++;
    }

    LINFO("Error count for %s is %d",inCompImage,errorCount[i]);
    LINFO("Total count for %s is %d",inCompImage,totCount[i]);
    LINFO("Error Ratio is %f",errorRatio[i]);
    LINFO("Euclidian distance is %f", eucDist[i]);
    LINFO("Regression %f",regression[i]);
    //sum stats
    totEucDist += eucDist[i];
    totalErrorRatio += errorRatio[i];
    totReg += regression[i];
    //find max and min values
    if(eucDist[i] > maxEuc){maxEuc = eucDist[i];}
    if(eucDist[i] < minEuc){minEuc = eucDist[i];}
    if(errorRatio[i] > maxMean){maxMean = errorRatio[i];}
    if(errorRatio[i] < minMean){minMean = errorRatio[i];}
    if(regression[i] > maxReg){maxReg = regression[i];}
    if(regression[i] < minReg){minReg = regression[i];}
  }
  meanEucDist = totEucDist/imageCount;
  totalMeanError = totalErrorRatio/(imageCount-zeroCount);
  meanReg = totReg/imageCount;

  stdEucDist = dStats.findS(eucDist, meanEucDist);
  totalStdError = Stats.findS(errorRatio, totalMeanError);
  stdReg = Stats.findS(regression, meanReg);

  LINFO("Total Images Run %d",imageCount);
  LINFO("Total Error Ratio %f",totalErrorRatio);
  LINFO("Total Mean Error %f :S %f",totalMeanError,totalStdError);
  LINFO("Max %f Min %f",maxMean,minMean);
  LINFO("Total Mean Euclidian Distance %f :S %f",meanEucDist,stdEucDist);
  LINFO("Max %f Min %f",maxEuc,minEuc);
  LINFO("Total Mean Regression %f :S %f",meanReg,stdReg);
  LINFO("Max %f Min %f",maxReg,minReg);
  LINFO("Zero Count %d",zeroCount);
}

void CINNICstatsRun::runPointAndFlood(readConfig &fileList,const char* param)
{
  std::ofstream outfile(statsFile.c_str(),std::ios::app);
  //while images to be read
  LINFO("STARTING Point and Flood");
  int N = 0;
  for(int i = 0; fileList.readFileTrue(i); i++)
  {
    maskImage.resize(compImage.getWidth(),compImage.getHeight(),true);
    N = i;
    fileList.readFileValueC(i);
    fileList.readFileValueNameC(i);
    // read the images
    tempImage1c = Raster::ReadRGB(fileList.readFileValueNameC(i), RASFMT_PNM);
    //LINFO("DONE");
    tempImage2 = Raster::ReadGray(fileList.readFileValueC(i), RASFMT_PNM);
    //LINFO("DONE");
    // Make sure they are good
    checkSize();
    // format them
    tempImage1cf = tempImage1c;
    testImage = luminance(tempImage1cf);
    inplaceAttenuateBorders(testImage, edgeAtten);
    compImage = tempImage2;
    salMap.resize(tempImage1c.getWidth(),tempImage1c.getHeight());
    outImageTemplate = compImage;
    outImageSource = testImage;
    //pixRGB.setGreen(128);
    //pixRGB.setBlue(0);
    //int setter = 255/pointsNum;

    if(preProcessPNF == 1)
      preProcess();

    //find salient points, circle them on any outimage, then flood out, repete
    pointAndFlood(fileList.readFileValueNameC(i),i,false);
  }

  for(int n = 0; n < pointsNum; n++)
  {
    strikeP[n] = (float)candPixels[n]/(float)realPixels[n];
    candPixels[n] = candPixels[n]/N;
    realPixels[n] = realPixels[n]/N;
    //strikeP[n] = strikeP[n];
    if(n > 0)
    {
      bernoulliP[n] = (1-bernoulliP[n-1])*strikeP[n] + bernoulliP[n-1];
    }
    else
    {
      bernoulliP[n] = strikeP[n];
    }
  }

  rankMean = rankTotal/N;
  rankSTD = Stats.findS(pointRank,rankMean);
  foundMean = foundTotal/N;
  foundSTD = Stats.findS(pointsFound,foundMean);

  LINFO("#################FINAL#################");
  LINFO("rank mean %f std %f",rankMean,rankSTD);
  LINFO("found mean %f std %f",foundMean,foundSTD);
  outfile << param << "\t";
  outfile << rankMean << "\t" << rankSTD << "\t";
  outfile << foundMean << "\t" << foundSTD << "\t";
  for(int i = 0; i < pointsNum; i++)
  {
    LINFO("AT rank %d = %d",i,pointRankNum[i]);
    outfile << pointRankNum[i] << "\t";
  }
  for(int i = 0; i <= pointsNum; i++)
  {
    LINFO("FOUND number %d = %d",i,pointsFoundNum[i]);
    outfile << pointsFoundNum[i] << "\t";
  }
  LINFO("General Stats:");
  for(int i = 0; i < pointsNum; i++)
  {
    LINFO("[%d] REAL %ld CANDIDATE %ld STRIKE %f BERNOULLI %f",i,realPixels[i]
          ,candPixels[i],strikeP[i],bernoulliP[i]);
    outfile << realPixels[i] << "\t" << candPixels[i] << "\t"
            << strikeP[i] << "\t" << bernoulliP[i] << "\t";
  }

  outfile << "\n";
  outfile.close();

}

void CINNICstatsRun::randomMatch(float *likelyhood,
                                 long *posRegionCount, long *totalCount)
{
  long pRC = 0, tC = 0;
  float lh;

  for(int i = 0; i < compImage.getWidth(); i++)
  {
    for(int j = 0; j < compImage.getHeight(); j++)
    {
      if(maskImage.getVal(i,j) < 10.0F)
      {
        tC++;
        if(compImage.getVal(i,j) > rankThresh)
        {
          pRC++;
        }
      }
    }
  }
  lh = pRC/tC;
  *posRegionCount = pRC;
  *totalCount = tC;
  *likelyhood = lh;
}

void CINNICstatsRun::preProcess()
{
  if(qualityImage == 0)
  {
    compImage = rescale(compImage,unEvenDec,unEvenDec);
    //Raster::VisuFloat(compImage, FLOAT_NORM_0_255, sformat("bar2.%d.pgm",i));
    compImage = lowPass5(compImage);
    // create some levity for my bad drawing

    for(int x = 0; x < compImage.getWidth(); x++)
    {
      for(int y = 0; y < compImage.getHeight(); y++)
      {
        if(compImage.getVal(x,y) < highThresh){compImage.setVal(x,y,255.0F);}
        else{compImage.setVal(x,y,0.0F);}
      }
    }
    compImage =\
      rescale(compImage,testImage.getWidth(),testImage.getHeight());

    //compImage = lowPass9(compImage);
    //compImage = lowPass5(compImage);

    // chop "black" values below chopVal (i.e. 31)
    for(int x = 0; x < compImage.getWidth(); x++)
    {
      for(int y = 0; y < compImage.getHeight(); y++)
      {
        if(compImage.getVal(x,y) < chopVal){compImage.setVal(x,y,0.0F);}
        if(testImage.getVal(x,y) < chopVal){testImage.setVal(x,y,0.0F);}
      }
    }
  }
  else
  {
    for(int x = 0; x < compImage.getWidth(); x++)
    {
      for(int y = 0; y < compImage.getHeight(); y++)
      {
        if(compImage.getVal(x,y) < qChop){compImage.setVal(x,y,0.0F);}
      }
    }
  }
}

void CINNICstatsRun::checkSize()
{
  if(doResize == 0)
  {
    ASSERT(tempImage1c.getWidth() == tempImage2.getWidth());
    ASSERT(tempImage1c.getHeight() == tempImage2.getHeight());
  }
  else
  {
    if(tempImage1c.getWidth() != tempImage2.getWidth())
    {
      if(tempImage1c.getHeight() != tempImage2.getHeight())
      {
        tempImage2 =\
          rescale(tempImage2,tempImage1c.getWidth(),tempImage1c.getHeight());
      }
      else
      {
        ASSERT("skew detected, cannot resize");
      }
    }
  }
}

PixRGB<float> CINNICstatsRun::colorTable(int i)
{
  PixRGB<float> pix;
  pix.setRed(0.0F);
  pix.setGreen(0.0F);
  pix.setBlue(0.0F);
  switch(i)
  {
  case 0: //red
    pix.setRed(255.0F);
    break;
  case 1: //orange
    pix.setGreen(128.0F);
    pix.setRed(255.0F);
    break;
  case 2: //yellow
    pix.setRed(255.0F);
    pix.setGreen(255.0F);
    break;
  case 3: //green
    pix.setGreen(255.0F);
    break;
  case 4: //blue
    pix.setBlue(255.0F);
    break;
  case 5: //violet(ish)
    pix.setRed(255.0F);
    pix.setBlue(255.0F);
    break;
  default: //white
    pix.setRed(255.0F);
    pix.setGreen(255.0F);
    pix.setBlue(255.0F);
  }
  return pix;
}

void CINNICstatsRun::pointAndFloodImage(Image<float> test_image,
                                        Image<float> sal_map
                                        ,int points,char* filename,
                                        float floodv,float floodt)
{
  monoColor = 0;
  pointsNum = points;
  useDrawDisk = 0;
  floodThresh = floodt;
  floodVal = floodv;
  circRad = 15;
  pointAndFloodImage(test_image,sal_map,filename);
}

void CINNICstatsRun::pointAndFloodImage(Image<float> test_image,
                                        Image<float> sal_map
                                        ,char* filename)
{
  LINFO("running POINT and FLOOD for %s",filename);
  point2D = new Point2D<int>;
  testImage = sal_map;
  maskImage.resize(sal_map.getWidth(),sal_map.getHeight(),true);
  outImageTemplate = test_image;
  outImageSource = sal_map;
  salMap.resize(sal_map.getWidth(),sal_map.getHeight(),true);
  pointAndFlood(filename,0,true);
}

void CINNICstatsRun::pointAndFlood(const char* filename,int i,bool standalone)
{
  Image<float> storeImage;
  for(int n = 0; n < pointsNum; n++)
  {
    if(!(standalone))
    {
      float strike = 0.0F;
      long cand = 0 ,real = 0;
      randomMatch(&strike,&cand,&real);
      candPixels[n] += cand;
      realPixels[n] += real;
    }

    float maxVal;
    findMax(testImage, *point2D,maxVal);
    salMap.setVal(*point2D,255.0F);
    //LINFO("PixelVal %f",compImage.getVal(*point2D));
    if(!(standalone))
    {
      if(compImage.getVal(*point2D) > rankThresh)
      {
        pointsFound[i]++;   //count up
        if(pointRank[i] == -1)
        {
          pointRank[i] = n; //set rank
          rankTotal += n;
          pointRankNum[(int)pointRank[i]]++;
          LINFO("Setting %d as rank %f",i,pointRank[i]);
        }
      }
    }
    //pixRGB.setRed(n*setter);
    pixRGB = colorTable(n);
    if(n > 0)
    {
      Point2D<int> *oldPoint = new Point2D<int>(ii,jj);
      //LINFO("P %d,%d %d,%d",point2D->i,point2D->j,oldPoint->i,oldPoint->j);
      if(monoColor != 1)
      {
        drawArrow(outImageTemplate, *oldPoint,*point2D,pixRGB);
        drawArrow(outImageSource, *oldPoint,*point2D,pixRGB);
      }
      else
      {
        pixRGB = colorTable(-1);
        drawArrow(outImageTemplate, *oldPoint,*point2D,pixRGB,2);
        drawArrow(outImageSource, *oldPoint,*point2D,pixRGB,2);
      }
    }
    ii = point2D->i;
    jj = point2D->j;
    if(monoColor != 1)
    {
      drawCircle(outImageTemplate, *point2D,circRad,pixRGB);
      drawCircle(outImageSource, *point2D,circRad,pixRGB);
    }
    else
    {
      //set white (default switch)
      pixRGB = colorTable(-1);
      drawCircle(outImageTemplate, *point2D,circRad,pixRGB,2);
      drawCircle(outImageSource, *point2D,circRad,pixRGB,2);
    }
    if(useDrawDisk != 1)
    {
      flood(testImage, storeImage,*point2D,floodThresh,floodVal);
      testImage -= storeImage;
    }
    else
    {
      drawDisk(testImage, *point2D,diskSize,0.0F);
      drawDisk(maskImage, *point2D,diskSize,255.0F);
    }
    //Raster::VisuFloat(testImage,0,"testimage.pgm");
  }
  if(!(standalone))
  {
    foundTotal += (int)pointsFound[i];
    pointsFoundNum[(int)pointsFound[i]]++;
    LINFO("Number Found %f / %d",pointsFound[i],pointsNum);
  }
  LINFO("writing %s.salPoint",filename);

  Image<PixRGB <byte> > thisBytes = outImageTemplate;
  Raster::WriteRGB(thisBytes,sformat("%s.salPoint.ppm",filename));
  thisBytes = outImageSource;
  Raster::WriteRGB(thisBytes,sformat("%s,salPointSource.ppm",filename));
}

float CINNICstatsRun::polatSagi2AFC(Image<float> targetImage,
                                    Image<float> notargetImage)
{
  ASSERT(targetImage.getHeight() == notargetImage.getHeight());
  ASSERT(targetImage.getWidth() == notargetImage.getWidth());
  int Xcenter = (targetImage.getWidth()/2)+centerOffset;
  int Ycenter = (targetImage.getHeight()/2)+centerOffset;
  mu1 = PSError + targetImage.getVal(Xcenter,Ycenter);
  mu2 = PSError + notargetImage.getVal(Xcenter,Ycenter);
  float sigma1 = sqrt(mu1);
  float sigma2 = sqrt(mu2);
  return Stats.getErrorGGC_2AFC(mu1,mu2,sigma1,sigma2);
}

float CINNICstatsRun::getMu1()
{
  return mu1;
}

float CINNICstatsRun::getMu2()
{
  return mu2;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
