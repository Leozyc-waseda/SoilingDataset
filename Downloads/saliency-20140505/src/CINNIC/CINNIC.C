/*!@file CINNIC/CINNIC.C CINNIC classes */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNIC.C $
// $Id: CINNIC.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "CINNIC/CINNIC.H"

#include "Image/ColorOps.H"
#include "Image/CutPaste.H"    // for inplacePaste()
#include "Image/Kernels.H"     // for gaborFilter2()
#include "Image/MathOps.H"
#include "Image/Transforms.H"
#include "Util/Timer.H"

#include <cmath>
#include <fstream>


// ######################################################################
//! Brute force, super inefficient 2D convolution (truncated filter boundary)
/*! check for zero pixels and skip */
template <class T> static
Image<typename promote_trait<T, float>::TP>
convolveCleanZero(const Image<T>& src, const Image<float>& filter)
{
  return convolveCleanZero(src, filter.getArrayPtr(),
                       filter.getWidth(), filter.getHeight());
}

// ######################################################################
//! Brute force, super inefficient 2D convolution (truncated filter boundary)
/*! check for zero pixels and skip */
template <class T> static
Image<typename promote_trait<T, float>::TP>
convolveCleanZero(const Image<T>& src, const float* filter,
              const int Nx, const int Ny)
{
  ASSERT(src.initialized()); ASSERT((Nx & 1) && (Ny & 1));
  const int w = src.getWidth(), h = src.getHeight();
  // promote the source image to float if necessary, so that we do the
  // promotion only once for all, rather than many times as we access
  // the pixels of the image; if no promotion is necessary, "source"
  // will just point to the original data of "src" through the
  // copy-on-write/ref-counting behavior of Image:
  typedef typename promote_trait<T, float>::TP TF;
  const Image<TF> source = src;
  Image<TF> result(w, h, NO_INIT);
  typename Image<TF>::const_iterator sptr = source.begin();
  typename Image<TF>::iterator dptr = result.beginw();

  int kkk = Nx * Ny - 1;
  int Nx2 = (Nx - 1) / 2, Ny2 = (Ny - 1) / 2;

  const TF zero(0);

  // very inefficient implementation; one has to be crazy to use non
  // separable filters anyway...
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i)
      {
        if((src.getVal(i,j) > 0.001F) || (src.getVal(i,j) < -0.001F))
        {
          TF sum = TF(); float sumw = 0.0F;
          for (int kj = 0; kj < Ny; ++kj)
          {
            int kjj = kj + j - Ny2;
            if (kjj >= 0 && kjj < h)
              for (int ki = 0; ki < Nx; ++ki)
              {
                int kii = ki + i - Nx2;
                if (kii >= 0 && kii < w)
                {
                  float fil = filter[kkk - ki - Nx*kj];
                  sum += sptr[kii + w * kjj] * fil;
                  sumw += fil;
                }
              }
          }
          *dptr++ = sum / sumw;
        }
        else
          *dptr++ = zero;
      }
  return result;
}


// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

void CINNIC::viewNeuronTemplate(ContourNeuronCreate<float> &NeuronTemplate
                          ,readConfig &config)
{

  max = 0;
  min = 0;
  avg = 0;
  float maxhold = 0,minhold = 0,avghold = 0;
  int n = AnglesUsed*AnglesUsed;

  for (int i = 0; i < AnglesUsed; i++)
  {
    for (int j = 0; j < AnglesUsed; j++)
    {
      output[i][j] = Image<float>(XSize+1,YSize+1, NO_INIT);
      for (int k = 0; k <= XSize; k++)
      {
        for (int l = 0; l <= YSize; l++)
        {
          output[i][j].setVal(k,l
            ,NeuronTemplate.FourDNeuralMap[i][j][k][l].angABD);
        }
      }
      getMinMaxAvg(output[i][j], min,max,avg);
      if(min < minhold){minhold = min;} //store total max value
      if(max > maxhold){maxhold = max;} //store total min value
      avghold += avg; //add to avg just for shits and grins
    }
  }
  Image<PixRGB<float> > theImage;
  theImage.resize(((XSize+1)*AnglesUsed),((YSize+1)*AnglesUsed));

  for (int i = 0; i < AnglesUsed; i++)
  {
    for (int j = 0; j < AnglesUsed; j++)
    {
      Poutput[i][j] = Image< PixRGB<float> >((XSize+1),(YSize+1), NO_INIT);
      Poutput[i][j] = normalizeRGPolar(output[i][j], max,min);
      inplacePaste(theImage, Poutput[i][j],Point2D<int>((i*XSize),(j*YSize)));
      //Raster::VisuRGB(Poutput[i][j], sformat("mask_out.%f.%f.ppm"
      //             ,NeuralAngles[i],NeuralAngles[j]));

    }
  }
  Raster::VisuRGB(theImage,"ANGLE_IMAGE.ppm");
  avg = avghold/n;
}

//*****************************************************************

void CINNIC::convolveTest(ContourNeuronCreate<float> &NeuronTemplate
                           ,readConfig &config, Image<float> &testImage)
{
  float total = 0.0F;
  float pos = 0.0F;
  float neg = 0.0F;
  int InspX, InspY;
  Image<float> output;
  output.resize(testImage.getWidth(),testImage.getHeight());
  for(int x = 0; x < testImage.getWidth(); x++)
  {
    for(int y = 0; y < testImage.getHeight(); y++)
    {
      output.setVal(x,y,0);
      for (int k = 0; k <= XSize; k++)
      {
        for (int l = 0; l <= YSize; l++)
        {
          float temp;
          InspX = x + (k-(int)XCenter);
          InspY = y - (l-(int)YCenter);
          if((InspX >= 0) && (InspY >= 0) && (InspX < testImage.getWidth()) &&
             (InspY < testImage.getHeight()))
          {
            temp = testImage.getVal(x,y)*
              testImage.getVal(InspX,InspY)*
              NeuronTemplate.FourDNeuralMap[0][0][k][l].angABD;
            float hold = output.getVal(x,y);
            hold += temp;
            output.setVal(x,y,hold);
            if(temp > 0)
              pos += temp;
            else
              neg += temp;
            total += temp;
          }
        }
      }
    }
  }
  LINFO("TOTAL VALUE IS %f, Pos %f, Neg %f",total,pos,neg);
  float pixTotal = 0.0F;
  for(int y = 0; y < testImage.getHeight(); y++)
  {
    for(int x = 0; x < testImage.getWidth(); x++)
    {
      if(output.getVal(x,y) > 0)
      {
        printf("%f ",output.getVal(x,y));
        pixTotal += output.getVal(x,y);
      }
      else
      {
        printf("%f ",output.getVal(x,y));
      }
    }
    printf("\n");
  }
    LINFO("TOTAL PIX VALUE IS %f",pixTotal);
}

//*****************************************************************

void CINNIC::configLoad(readConfig &config)
{
  iterations = (int)config.getItemValueF("iterations");
  edge = (int)config.getItemValueF("edgeAtten");
  Amp = (int)config.getItemValueF("edgeAmp");
  dev = config.getItemValueF("devMult");
  cheatVal = (int)config.getItemValueF("cheatVal");
  Gnum = (int)config.getItemValueF("Gnum");
  cheatNum = (int)config.getItemValueF("cheatNum");
  logto = config.getItemValueC("logOutDir");
  saveto = config.getItemValueC("imageOutDir");
  dumpImage = (int)config.getItemValueF("dumpImage");
  redOrder = (int)config.getItemValueF("redOrder");
  lPass = (int)config.getItemValueF("lPass");
  reduction = (int)config.getItemValueF("reduction");
  groupSize = (int)config.getItemValueF("groupSize");
  scalesNumber = (int)config.getItemValueF("scalesNumber");
  GroupTop = config.getItemValueF("GroupTop");
  lastIterOnly = (int)config.getItemValueF("lastIterOnly");
  addNoise = (int)config.getItemValueF("addNoise");
  preOrientFilterNoise = config.getItemValueF("preOrientFilterNoise");
  storeArraySize = (int)config.getItemValueF("storeArraySize");
  BiasDiff = config.getItemValueF("BiasDiff");
  GridBias = config.getItemValueF("GridBias");
  doNerdCam = (int)config.getItemValueF("doNerdCam");
  doBias = (int)config.getItemValueF("doBias");
  doGaborFilter = (int)config.getItemValueF("doGaborFilter");
  Gstddev = config.getItemValueF("Gstddev");
  Gperiod = config.getItemValueF("Gperiod");
  Gphase = config.getItemValueF("Gphase");
  GsigMod = config.getItemValueF("GsigMod");
  Gamplitude = config.getItemValueF("Gamplitude");
  doTableOnly = (int)config.getItemValueF("doTableOnly");
  compGain = config.getItemValueF("compGain");
  scale1 = config.getItemValueF("scale1");
  scale2 = config.getItemValueF("scale2");
  scale3 = config.getItemValueF("scale3");
}

/*! This is a simple ramp (up then down) function to bias the
  oriented images from vertical
  filter more then horizontal filter. This is to adjust and create a linear
  function observed on Polat and Sagi Spat. Vis. 1994
  ALSO: add a bias for 45 degree angles due to the effect of the images
  grid shape.
*/

void CINNIC::findNeuralAnglesBias()
{
  int lastAngle = 0;
  for(int i = 0; i < AnglesUsed/2; i++)
  {
    NeuralAnglesBias[i] =
      1-((BiasDiff/(AnglesUsed/2))*((AnglesUsed/2)-i));
    LINFO("BIAS1 %f",NeuralAnglesBias[i]);
    if(i <= (AnglesUsed/4))
      NeuralAnglesBias[i] += (GridBias/(AnglesUsed/4))*i;
    else
      NeuralAnglesBias[i] += (GridBias/(AnglesUsed/4))*((AnglesUsed/2)-i);
    LINFO("BIAS2 %f",NeuralAnglesBias[i]);
    lastAngle++;
  }
  for(int i = lastAngle; i < AnglesUsed; i++)
  {
    NeuralAnglesBias[i] = 1-((BiasDiff/(AnglesUsed/2))*(i-(AnglesUsed/2)));
    LINFO("BIAS1 %f",NeuralAnglesBias[i]);
    if(i <= ((3*AnglesUsed)/4))
      NeuralAnglesBias[i] += (GridBias/(AnglesUsed/4))*(i-(AnglesUsed/2));
    else
      NeuralAnglesBias[i] += (GridBias/(AnglesUsed/4))*(AnglesUsed-i);
    LINFO("BIAS2 %f",NeuralAnglesBias[i]);
  }
}

//*****************************************************************
//*****************************************************************
// START HERE
//*****************************************************************
//*****************************************************************

void CINNIC::RunSimpleImage(ContourNeuronCreate<float> &NeuronTemplate,
                                   Image<byte> &input, readConfig &config)
{
  origX = input.getWidth();origY = input.getHeight();
  orientComposite.resize(256,256,0.0F);
  Original = input;
  // Loads the config file for CINNIC params
  LINFO("LOAD CONFIG");
  configLoad(config);
  std::ofstream outfile("CINNICtimer.log",std::ios::app);
  Timer tim;
  tim.reset();
  // resize and do orientation filtering on input image
  LINFO("PRE-PROCESS IMAGE");
  preProcessImage(NeuronTemplate,input,config);
  // set up image vectors and group sising
  LINFO("PRE-IMAGE");
  preImage(VFinput,config,input.getWidth(),input.getHeight());
  // actaully run the image for x number of iterations
  LINFO("RUN IMAGE");
  runImage(NeuronTemplate,VFinput,config,groupTopVec);
  uint64 t0 = tim.get();
  // get and output results
  LINFO("GET RESULTS");
  getResults(config);

  printf("\n*************************************\n");
  printf("Time to convolve, %llums seconds\n",t0);
  printf("*************************************\n\n");
  outfile << t0 << "\t";
  outfile.close();
}

//*****************************************************************

void CINNIC::preProcessImage(ContourNeuronCreate<float> &NeuronTemplate
                      ,Image<byte> &input,readConfig &config)
{
  if(doGaborFilter == 0)
  {
    if((input.getWidth() > 256) || (input.getHeight() > 256))
      input = rescale(input,256,256);
  }

  floatImage = input;
  if(addNoise == 1)
  {
    LINFO("ADDING noise %f",preOrientFilterNoise);
    inplaceAddBGnoise2(input, preOrientFilterNoise);
    Raster::VisuGray(input, sformat("post_noise_image_in_%d.pgm", 0));
  }

  // Resize and filter image into oriented images

  if(doBias == 1)
    findNeuralAnglesBias();
  Finput.resize(AnglesUsed,input);
  VFinput.resize(scalesNumber,Finput);
  for(int i = 0; i < AnglesUsed; i++)
  {
    Finput[i] = Image<float>(input.getDims(),ZEROS);
    Finput[i] = input;  // convert input to float
    Image<float> i2 = Finput[i];
    if(redOrder == 0)
    {
      LINFO("REDUCTION USING decXY");
      while(Finput[i].getWidth() > reduction)
      {
        Finput[i] = decXY(Finput[i]);
      }
    }

    if(lPass == 0)
    {
      i2 = lowPass9(i2);
      Finput[i] -= i2;
    }
    if(lPass == 1)
    {
      i2 = lowPass5(i2);
      Finput[i] -= i2;
    }
    if(lPass == 2)
    {
      i2 = lowPass3(i2);
      Finput[i] -= i2;
    }

    if(doGaborFilter == 0)
    {
      float mpi =  M_PI / Gnum;
      Finput[i] = orientedFilter(Finput[i],mpi,NeuralAngles[i]);
      inplaceAttenuateBorders(Finput[i], edge);
    }
    else
    {
      LINFO("Using gabor filter");
      // use this is you want accuracy, but not speed
      Gtheta = NeuralAngles[i];
      Image<float> filter;
      filter = gaborFilter2<float>(Gstddev,
                                   Gperiod,Gphase,Gtheta,
                                   GsigMod,Gamplitude);
      // NOTE: added an extra 'i' at the end of sformat(), because
      // there were only three format parameters and the final %d was
      // unmatched.
      Raster::VisuFloat((filter+128),0,
                        sformat("%s%s.filter_bias.image_%f_%d.pgm",saveto
                                ,savefilename,NeuralAngles[i], i));
      Finput[i] = convolveCleanZero(Finput[i],filter);
      // NOTE: added an extra 'i' at the end of sformat(), because
      // there were only three format parameters and the final %d was
      // unmatched.
      Raster::VisuFloat((Finput[i]+128),0,
                        sformat("%s%s.pre_bias.image_%f_%d.pgm",saveto
                                ,savefilename,NeuralAngles[i], i));
      rescale(Finput[i],256,256);
    }
    Image<float> tempSizer = Finput[i];
    rescale(tempSizer,256,256); // <--- FIX HERE
    orientComposite += tempSizer;
    if(doBias == 1)
    {
      LINFO("NEURALANGLEBIAS %f",NeuralAnglesBias[i]);
      for(int x = 0; x < Finput[i].getWidth(); x++)
      {
        for(int y = 0; y < Finput[i].getHeight(); y++)
        {
          Finput[i].setVal(x,y,(Finput[i].getVal(x,y)*NeuralAnglesBias[i]));
        }
      }
    }

    // resize images into different scales

    if(redOrder == 1)
    {
      int WHold = Finput[i].getWidth();
      int HHold = Finput[i].getHeight();
      while((WHold >= ImageSizeX) || (HHold >= ImageSizeY))
      {
        WHold = WHold/2;
        HHold = HHold/2;
      }
      for(int s = 0; s < scalesNumber; s++)
      {
        VFinput[s][i] = Finput[i];
        LINFO("Image Scale %d %d", WHold, HHold);
        VFinput[s][i] = rescale(VFinput[s][i], WHold,HHold);
        WHold = WHold/2;
        HHold = HHold/2;
        //uncomment this line to see all scale maps
        //Raster::VisuFloat(VFinput[s][i], FLOAT_NORM_0_255, sformat("%s%s.image_%f_%d_%d.pgm"
        //  ,saveto,savefilename,NeuralAngles[i],s));
      }
    }

    //uncomment these three lines to view the oriented filter output
    //Raster::WriteGray(Finput[i],
    //          sformat("%s%s.image_%f_%d.pgm",saveto,savefilename
    //                  ,NeuralAngles[i]));
    SY = Finput[i].getHeight();
    SX = Finput[i].getWidth();
    mean = ::mean(Finput[i]);
    std = ::stdev(Finput[i]);
    getMinMaxAvg(Finput[i], min,max,avg);
  }
  if(doGaborFilter == 1)
  {
    if((input.getWidth() > 256) || (input.getHeight() > 256))
      input = rescale(input,256,256);
  }
}

//*****************************************************************

void CINNIC::preImage(std::vector< std::vector<Image<float> > > &input
                      ,readConfig &config,int sizeX, int sizeY)
{
  orientComposite = (orientComposite/AnglesUsed)*compGain;
  SIZEX = sizeX; SIZEY = sizeY;
  Image<float> foo(SIZEX,SIZEY,ZEROS);
  group.resize(scalesNumber,foo);
  groupCount.resize(scalesNumber);
  groupTopVec.resize(scalesNumber);

  contourRun *CR = new contourRun[scalesNumber]();

  RN = CR;

  combinedSalMap.resize(iterations,foo);
  combinedSalMapMax.resize(iterations,foo);
  combinedSalMapMin.resize(iterations,foo);
  //LINFO("Resizing combined sal map");
  for(int i = 0; i < iterations; i++)
  {
    combinedSalMap[i] = Image<byte>(SIZEX,SIZEY,ZEROS);
    combinedSalMapMax[i] = Image<byte>(SIZEX,SIZEY,ZEROS);
    combinedSalMapMin[i] = Image<byte>(SIZEX,SIZEY,ZEROS);
    //LINFO("RESIZED %d:%d at %d",SIZEX,SIZEY,i);
  }

  int GS = groupSize;
  int size = storeArraySize;

  // size and label groups

  for(int s = 0; s < scalesNumber; s++)
  {
    RN[s].setArraySize(size);
    size = size/4;
    if(s > 0)
    {
      GS = GS/2;
    }
    groupCount[s] = 0;
    groupTopVec[s] = GroupTop;
    GroupTop = GroupTop/4;

    group[s] = Image<float>(input[s][0].getDims(),ZEROS);
    int Hmod,Wmod;
    int Wtemp = input[s][0].getWidth()/GS;
    if((Wmod = (input[s][0].getWidth()%GS)) != 0){Wtemp++;}
    int Htemp = input[s][0].getHeight()/GS;
    if((Hmod = (input[s][0].getHeight()%GS)) != 0){Htemp++;}

    for(int x = 0; x < Wtemp; x++)
    {
      for(int y = 0; y < Htemp; y++)
      {
        for(int i = 0; i < GS; i++)
        {
          for(int j = 0; j < GS; j++)
          {
            int intx = i+(GS*x);
            int inty = j+(GS*y);
            if((intx < group[s].getWidth()) && (inty < group[s].getHeight()))
            {
              group[s].setVal(intx,inty,(float)groupCount[s]);
            }
          }
        }
        groupCount[s]++;
      }
    }
  }
};

//*****************************************************************

void CINNIC::runImage(ContourNeuronCreate<float> &NeuronTemplate,
                      std::vector< std::vector<Image<float> > > &input
                      ,readConfig &config, std::vector<float> &GTV)
{
  //SIZEX = SIZEX * 2; // <--- FIX HERE
  //SIZEY = SIZEY * 2; // <--- FIX HERE
  Image<float> floatTemp;
  Image<float> floatTemp2;
  Image<byte> floatByte;
  Image<float> temp;
  Image<float> tempT;
  Image<float> temp2;
  Image<float> tempMax;
  Image<float> tempMin;
  char avgFileName[100];
  sprintf(avgFileName,"%s%s.table.out.txt"
          ,saveto,savefilename);
  std::ofstream avgFile(avgFileName,std::ios::out);

  //optimize this dammit!!!!
  for(int t = 0; t < iterations; t++)
  {
    floatTemp2 = Image<float>(SIZEX,SIZEY,ZEROS);
    temp2 = Image<float>(SIZEX,SIZEY,ZEROS);
    tempMax = Image<float>(SIZEX,SIZEY,ZEROS);
    tempMin = Image<float>(SIZEX,SIZEY,ZEROS);
    for(int s = 0; s < scalesNumber; s++)
    {
      // weighted average bias for scales output
      float bias;
      switch(s)
      {
      case 0:
        bias = scale1;
        break;
      case 1:
        bias = scale2;
        break;
      case 2:
        bias = scale3;
        break;
      default:
        bias = 1.0F;
        break;
      }

      // **********************
      // ** Do the fun stuff **
      // **********************
      // see: contourRun.*
      RN[s].contourRunMain(VFinput[s],NeuronTemplate,config,group[s],
                        groupCount[s],t,GTV[s]);
      // **********************
      // Post process results for this iteration

      tempT = RN[s].getSMI(t);
      temp.resize(tempT.getWidth(),tempT.getHeight());

      // bump the sal map over a pixel
      for(int i = 0; i < tempT.getWidth(); i++)
      {
        for(int j = 0; j < tempT.getHeight(); j++)
        {
          if((j == 0) || (i == 0))
          {
            temp.setVal(i,j,0);
          }
          else
          {
            temp.setVal(i,j,tempT.getVal(i-1,j-1));
          }
        }
      }

      floatTemp = rescale(temp,SIZEX,SIZEY);
      temp = rescale(temp,SIZEX,SIZEY);


      //set a bias for scales if needed in avg map
      for(int i = 1; i < temp.getWidth(); i++)
      {
        for(int j = 1; j < temp.getHeight(); j++)
        {
          float foo = temp.getVal(i,j)*bias;
          temp.setVal(i,j,foo);
        }
      }
      if(t == (iterations - 1))
      {
        for(int i = 0; i < floatTemp.getWidth(); i++)
        {
          for(int j = 0; j < floatTemp.getHeight(); j++)
          {
            float foo = floatTemp.getVal(i,j)*bias;
            floatTemp.setVal(i,j,foo);
          }
        }
        floatByte = floatTemp;
        if(doTableOnly == 0)
        {
          Raster::WriteGray(floatByte,
                            sformat("%s%s.minimum.out.%d.pgm"
                                    ,saveto,savefilename,s));
        }
        floatTemp2 += floatTemp;
      }

      //set min or max for each scale
      //reset min matrix
      for(int i = 0; i < temp.getWidth(); i++)
      {
        for(int j = 0; j < temp.getHeight(); j++)
        {
          tempMin.setVal(i,j,255);
        }
      }

      for(int i = 0; i < temp.getWidth(); i++)
      {
        for(int j = 0; j < temp.getHeight(); j++)
        {
          if(temp.getVal(i,j) > tempMax.getVal(i,j))
          {
            //set max pixel
            tempMax.setVal(i,j,temp.getVal(i,j));
          }
          if(temp.getVal(i,j) < tempMin.getVal(i,j))
          {
            tempMin.setVal(i,j,temp.getVal(i,j));
          }
        }
      }
      temp2+=temp;
    }

    //divide image by group numbers so that the output fits in 256 greys
    double mint;

    if(t == (iterations - 1))
    {
      avgFile << floatTemp2.getWidth() << "\t" << floatTemp2.getHeight() << "\n";
      for(int i = 0; i < floatTemp2.getWidth(); i++)
      {
        for(int j = 0; j < floatTemp2.getHeight(); j++)
        {
          mint = floatTemp2.getVal(i,j)/scalesNumber;
          avgFile << mint << "\t";
          floatTemp2.setVal(i,j,mint);
        }
        avgFile << "\n";
      }
      floatByte = floatTemp2;
      if(doTableOnly == 0)
      {
        Raster::WriteGray(floatByte,
                          sformat("%s%s.minimumSize.out.pgm"
                                  ,saveto,savefilename));
      }
    }

    for(int i = 0; i < temp2.getWidth(); i++)
    {
      for(int j = 0; j < temp2.getHeight(); j++)
      {
        mint = temp2.getVal(i,j)/scalesNumber;
        temp2.setVal(i,j,mint);
      }
    }
    int xx = temp2.getWidth();
    int yy = temp2.getHeight();
    combinedSalMap[t] = temp2;
    combinedSalMapMax[t] = tempMax;
    combinedSalMapMin[t] = tempMin;
    if(((lastIterOnly == 0) || (t == (iterations - 1))) && (doTableOnly == 0))
    {
      //Raster::VisuGray(combinedSalMap[t],sformat("combined_map_%d.pgm",t));
      if(doNerdCam == 1)
      {
        combinedSalMap[t] = rescale(combinedSalMap[t],origX,origY);
        Image<float> nerd;
        nerd = combinedSalMap[t];
        char savename[100];
        sprintf(savename,"%s%s.sal.out.%d.%d.%d.%d",saveto,savefilename
                ,0,t,xx,yy);
        cinnicStats.pointAndFloodImage(Original,nerd,6,savename,24.0F,48.0F);
        inplaceNormalize(combinedSalMap[t], byte(0), byte(255));
      }

      Raster::WriteRGB(Image<PixRGB<byte> >(combinedSalMap[t])
                       ,sformat("%s%s.avg.out.%d.%d.%d.%d.ppm"
                                    ,saveto,savefilename,0,t,xx,yy));

      if(doNerdCam == 0)
      {
        //Raster::WriteRGB(Image<PixRGB<byte> >(combinedSalMap[t]+orientComposite)
        //              ,sformat("%s%s.Compavg.out.%d.%d.%d.%d.ppm"
        //             ,saveto,savefilename,0,t,xx,yy));

        //Raster::VisuGray(combinedSalMapMax[t],sformat("max_map_%d.pgm",t));
        Raster::WriteRGB(Image<PixRGB<byte> >(combinedSalMapMax[t])
                         ,sformat("%s%s.max.out.%d.%d.%d.%d.ppm"
                                      ,saveto,savefilename,0,t,xx,yy));

        //Raster::VisuGray(combinedSalMapMin[t],sformat("min_map_%d.pgm",t));
        Raster::WriteRGB(Image<PixRGB<byte> >(combinedSalMapMin[t])
                         ,sformat("%s%s.min.out.%d.%d.%d.%d.ppm"
                                      ,saveto,savefilename,0,t,xx,yy));
        Image<float> floatTempT;
      }
    }
  }
  std::ofstream propFile("CINNIC2.prop",std::ios::out);
  propFile << "type avg\n";
  propFile << "type max\n";
  propFile << "type min\n";
  propFile.close();
  avgFile.close();
};

void CINNIC::getResults(readConfig &config)
{

  for(int i = 0; i < scalesNumber; i++)
  {
    LINFO("DUMPING ENERGY MAP");
    char holder[256];
    strcpy(holder,logto);
    RN[i].dumpEnergySigmoid(strcat(holder,"energy"),
                            savefilename,config,floatImage,i,scalesNumber);
    LINFO("Unique iterations at scale %d = %ld",i,RN[i].iterCounter);
  }
  LINFO("FINISHED");
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
