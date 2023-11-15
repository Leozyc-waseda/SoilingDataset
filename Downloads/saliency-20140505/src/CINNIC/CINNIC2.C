/*!@file CINNIC/CINNIC2.C CINNIC2 classes */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNIC2.C $
// $Id: CINNIC2.C 6191 2006-02-01 23:56:12Z rjpeters $
//

#include "CINNIC/CINNIC2.H"

#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/MathOps.H"
#include "Image/ShapeOps.H"

#include <cmath>

using std::vector;
using std::string;

// ############################################################
// ############################################################
// ##### ---CINNIC2---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

//*****************************************************************
CINNIC2_DEC
CINNIC2_CLASS::CINNIC2()
{
  CINuseFrameSeries = false;
}

//*****************************************************************
CINNIC2_DEC
CINNIC2_CLASS::~CINNIC2()
{}

//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINtoggleFrameSeries(bool toggle)
{
  CINuseFrameSeries = toggle;
}

//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINconfigLoad(readConfig &config)
{

  // Some of these do nothing, I am weeding them out as I can
  CINedgeAtten      = (INT)config.getItemValueF("edgeAtten");
  CINGnum           = (INT)config.getItemValueF("Gnum");
  CINlogto          = config.getItemValueS("logOutDir");
  CINsaveto         = config.getItemValueS("imageOutDir");
  CINlPass          = (INT)config.getItemValueF("lPass");
  CINgroupSize      = (INT)config.getItemValueF("groupSize");
  CINGroupTop       = config.getItemValueF("GroupTop");
  CINstoreArraySize = (INT)config.getItemValueF("storeArraySize");
  CINscaleBias[0]   = config.getItemValueF("scaleBias1");
  CINscaleBias[1]   = config.getItemValueF("scaleBias2");
  CINscaleBias[2]   = config.getItemValueF("scaleBias3");
  CINscaleSize[0]   = (INT)config.getItemValueF("scaleSize1");
  CINscaleSize[1]   = (INT)config.getItemValueF("scaleSize2");
  CINscaleSize[2]   = (INT)config.getItemValueF("scaleSize3");
  CINbaseSize       = (INT)config.getItemValueF("baseSize");
  CINuseMaps        = config.getItemValueB("useMaps");
}

//*****************************************************************
//*****************************************************************
// START HERE
//*****************************************************************
//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINrunSimpleImage(
                         const ContourNeuronCreate<FLOAT> &NeuronTemplate,
                         const char fileName[100], unsigned int frame,
                         Image<byte> &input,
                         readConfig &config)
{
  CINframe = frame;
  LINFO("FRAME %d",CINframe);
  Timer tim;
  tim.reset();
  int t1 = 0;
  int t2 = 0;
  int t3 = 0;
  int t0 = tim.get();  // to measure display time
  if(frame == 1)
  {
    LINFO("Read Config");
    CINconfigLoad(config);
    t1 = tim.get();
    t2 = t1 - t0; t3 = t2;
    LINFO("TIME: %d ms Slice: %d ms",t2,t3);

    LINFO("Init CINNIC");
    CINinitCINNIC(CINVFinput,config,input.getWidth(),input.getHeight());
    t1 = tim.get();
    t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
    LINFO("TIME: %d ms Slice: %d ms",t2,t3);
  }

  LINFO("Get Orient Filtered Image");
  if(CINuseMaps == false)
    CINgetOrientFiltered(input);
  else
    CINgetOrientFilteredMap(input);
  t1 = tim.get();
  t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
  LINFO("TIME: %d ms Slice: %d ms",t2,t3);

  LINFO("Get Scaled Images");
  CINgetScaled();
  t1 = tim.get();
  t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
  LINFO("TIME: %d ms Slice: %d ms",t2,t3);

  if(frame == 1)
  {
    LINFO("Compute Groups");
    CINcomputeGroups(CINVFinput);
    t1 = tim.get();
    t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
    LINFO("TIME: %d ms Slice: %d ms",t2,t3);
  }

  LINFO("Run Image");
  if(CINuseFrameSeries == false)
    CINrunImage(NeuronTemplate,CINVFinput,config,CINgroupTopVec);
  else
    CINrunImageFrames(NeuronTemplate,CINVFinput,config,CINgroupTopVec);
  t1 = tim.get();
  t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
  LINFO("TIME: %d ms Slice: %d ms",t2,t3);
  //CINgetResults(config);
}

//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINgetOrientFiltered(Image<byte> &input)
{
  CINorignalImageSizeX = input.getWidth();
  CINorignalImageSizeY = input.getHeight();
  Image<float> inputCopy = input;

  while((inputCopy.getWidth()  > CINbaseSize) ||
        (inputCopy.getHeight() > CINbaseSize))
  {
    inputCopy = decXY(inputCopy);
  }

  LINFO("(1) Image resized via decimation to %d x %d",
        inputCopy.getWidth(),inputCopy.getHeight());

  Image<FLOAT> i2 = inputCopy;

  if(CINlPass == 0)
  {
    i2 = lowPass9(i2);
    Raster::VisuFloat(i2, FLOAT_NORM_0_255, sformat("lowPass.out.pgm"));
    inputCopy -= i2;
  }
  else if(CINlPass == 1)
  {
    i2 = lowPass5(i2);
    Raster::VisuFloat(i2, FLOAT_NORM_0_255, sformat("lowPass.out.pgm"));
    inputCopy -= i2;
  }
  else if(CINlPass == 2)
  {
    i2 = lowPass3(i2);
    Raster::VisuFloat(i2, FLOAT_NORM_0_255, sformat("lowPass.out.pgm"));
    inputCopy -= i2;
  }

  for(INT i = 0; i < CINorientations; i++)
  {
    const FLOAT mpi =  M_PI / CINGnum;
    CINFinput[i] = orientedFilter(inputCopy,mpi,NeuralAngles[i]);
    inplaceAttenuateBorders(CINFinput[i], CINedgeAtten);
  }
}
//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINgetOrientFilteredMap(Image<byte> &input)
{
  CINorignalImageSizeX = input.getWidth();
  CINorignalImageSizeY = input.getHeight();
  Image<FLOAT> inputCopy = input;

  while((inputCopy.getWidth()  > CINbaseSize) ||
        (inputCopy.getHeight() > CINbaseSize))
  {
    inputCopy = decXY(inputCopy);
  }

  LINFO("(2) Image resized via decimation to %d x %d",
        inputCopy.getWidth(),inputCopy.getHeight());

  if(CINframe == 1)
  {
    readMatrix rm("lowPassKernel.mat");
    rm.echoMatrix();
    CINcMap.CMsmallNumber = 1.1F;
    CINcMap.CMinitVecSize = 1;
    CINcMap.CMkernel = rm.returnMatrixAsImage();
    CINcMap.CMorigImage = inputCopy;
    computeConvolutionMaps(CINcMap);
  }

  CINcMap.CMcopyImage(inputCopy);
  //Raster::Visu(CINcMap.CMstaticImage, true, false,
  //             sformat("staticImage.pgm"));
  Image<FLOAT> i2 = convolveWithMaps(CINcMap);
  //Raster::VisuGray(i2, sformat("lowPassImage.%06d.pgm",CINframe));
  inputCopy -= i2;

  for(INT i = 0; i < CINorientations; i++)
  {
    const FLOAT mpi =  M_PI / CINGnum;
    CINFinput[i] = orientedFilter(inputCopy,mpi,NeuralAngles[i]);
    inplaceAttenuateBorders(CINFinput[i], CINedgeAtten);
  }
}

//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINgetScaled()
{
  for(INT i = 0; i < CINorientations; i++)
  {
    // resize images INTo different scales

    INT WHold, HHold;
    for(INT s = 0; s < CINscales; s++)
    {
      if(CINFinput[i].getWidth() > CINFinput[i].getHeight())
      {
        const FLOAT ratio =
          ((float)CINscaleSize[s]/(float)CINFinput[i].getWidth());
        WHold = CINscaleSize[s];
        HHold = (int)(CINFinput[i].getHeight() * ratio);
      }
      else
      {
        const FLOAT ratio =
          ((float)CINscaleSize[s]/(float)CINFinput[i].getHeight());
        HHold = CINscaleSize[s];
        WHold = (int)(CINFinput[i].getWidth() * ratio);
      }
      //CINVFinput[s][i] = CINFinput[i];
      //LINFO("Image Scale %d %d", WHold, HHold);
      CINVFinput[s][i] = rescale(CINFinput[i], WHold,HHold);
      //uncomment this line to see all scale maps
      //Raster::VisuFloat(CINVFinput[s][i], FLOAT_NORM_0_255,
      //                  sformat("image.%f.%d.%06d.pgm",NeuralAngles[i],s,CINframe));
    }
  }
}

//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINinitCINNIC(const std::vector< std::vector<Image<FLOAT> > > &input,
                                  readConfig &config,
                                  const INT sizeX,
                                  const INT sizeY)
{
  Image<FLOAT> blankImage;
  Image<FLOAT> foo(sizeX,sizeY,ZEROS);

  //! resize and create contourRun2 for scales
  contourRun2<CINkernelSize,CINscales,
              CINorientations,CINiterations,
              FLOAT,INT> cont;

  CINcontourRun.resize(CINscales,cont);

  // Resize and filter image INTo oriented images
  CINFinput.resize(AnglesUsed,blankImage);
  CINVFinput.resize(CINscales,CINFinput);

  CINresults.resize(CINiterations,blankImage);
  CINIresults.resize(CINscales,CINresults);

  CINgroup.resize(CINscales,foo);
  CINgroupCount.resize(CINscales);
  CINgroupTopVec.resize(CINscales);
  //CINcontourRun.resize(CINscales);

  CINcombinedSalMap = Image<byte>(sizeX,sizeY,ZEROS);
};

//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINcomputeGroups(
                                const std::vector< std::vector<Image<FLOAT> > > &input)
{
  INT GS = CINgroupSize;
  INT size = CINstoreArraySize;

  // size and label groups

  for(unsigned short s = 0; s < CINscales; s++)
  {
    size = size/4;
    if(s > 0)
    {
      GS = GS/2;
    }
    CINgroupCount[s] = 0;
    CINgroupTopVec[s] = CINGroupTop;
    CINGroupTop = CINGroupTop/4;

    CINgroup[s] = Image<FLOAT>(input[s][0].getDims(),ZEROS);
    INT Hmod,Wmod;
    INT Wtemp = input[s][0].getWidth()/GS;
    if((Wmod = (input[s][0].getWidth()%GS)) != 0){Wtemp++;}
    INT Htemp = input[s][0].getHeight()/GS;
    if((Hmod = (input[s][0].getHeight()%GS)) != 0){Htemp++;}

    for(INT x = 0; x < Wtemp; x++)
    {
      for(INT y = 0; y < Htemp; y++)
      {
        for(INT i = 0; i < GS; i++)
        {
          for(INT j = 0; j < GS; j++)
          {
            const INT intx = i+(GS*x);
            const INT inty = j+(GS*y);
            if(   (intx < CINgroup[s].getWidth())
               && (inty < CINgroup[s].getHeight()))
            {
              CINgroup[s].setVal(intx,inty,
                                 static_cast<FLOAT>(CINgroupCount[s]));
            }
          }
        }
        CINgroupCount[s]++;
      }
    }
  }
};

//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINrunImage(
                          const ContourNeuronCreate<FLOAT> &NeuronTemplate,
                          const std::vector< std::vector<Image<FLOAT> > > &input,
                          readConfig &config,
                          const std::vector<FLOAT> &GTV)
{
  for(unsigned short t = 0; t < CINiterations; t++)
  {
    LINFO(">>>> ITERATION %d <<<<",t);
    CINresults[t].resize(CINVFinput[0][0].getWidth(),
                         CINVFinput[0][0].getHeight(),true);
    for(unsigned short s = 0; s < CINscales; s++)
    {
      LINFO("SCALE %d",s);
      LINFO("%d x %d",CINVFinput[s][0].getWidth()
            ,CINVFinput[s][0].getHeight());
      //LINFO("2 %d %d",CINgroup[s].getWidth(),CINgroup[s].getHeight());
      //LINFO("3 %d",GTV[s]);
      CINcontourRun[s].CONTtoggleFrameSeries(CINuseFrameSeries);
      CINcontourRun[s].CONTcontourRunMain(CINVFinput[s],NeuronTemplate,
                                          config,CINgroup[s],
                                          CINgroupCount[s],t,GTV[s]);
      CINIresults[s][t] = CINcontourRun[s].CONTgetSMI(t);
      CINresults[t] += rescale(CINIresults[s][t] * CINscaleBias[s],
                               CINresults[t].getWidth(),
                               CINresults[t].getHeight());
    }
    CINresults[t] /= CINscales;
    Image<float> resultsRescaled = rescale(CINresults[t],
                                           CINorignalImageSizeX,
                                           CINorignalImageSizeY);
    /*
    if(t < 10)
      Raster::Visu(resultsRescaled,
                   sformat("results.00%d.out.pgm",t));
    else if(t < 100)
      Raster::Visu(resultsRescaled,
                   sformat("results.0%d.out.pgm",t));
    else
      Raster::Visu(resultsRescaled,
                   sformat("results.%d.out.pgm",t));
    */

    Raster::VisuFloat(resultsRescaled, FLOAT_NORM_0_255,
                      sformat("results.%06d.out.pgm",t));
  }
};


//*****************************************************************
CINNIC2_DEC
void CINNIC2_CLASS::CINrunImageFrames(
                          const ContourNeuronCreate<FLOAT> &NeuronTemplate,
                          const std::vector< std::vector<Image<FLOAT> > > &input,
                          readConfig &config,
                          const std::vector<FLOAT> &GTV)
{
  INT iter = 0;
  if(CINframe == 1)
  {
    for(unsigned short t = 0; t < CINiterations; t++)
    {
      CINresults[t].resize(CINVFinput[0][0].getWidth(),
                           CINVFinput[0][0].getHeight(),true);
    }
  }

  for(unsigned short s = 0; s < CINscales; s++)
  {
    LINFO("SCALE %d",s);
    LINFO("%d x %d",CINVFinput[s][0].getWidth()
          ,CINVFinput[s][0].getHeight());
    //LINFO("2 %d %d",CINgroup[s].getWidth(),CINgroup[s].getHeight());
    //LINFO("3 %d",GTV[s]);
    CINcontourRun[s].CONTtoggleFrameSeries(true);
    CINcontourRun[s].CONTcontourRunFrames(CINVFinput[s],NeuronTemplate,
                                          config,CINgroup[s],
                                          CINgroupCount[s],CINframe,GTV[s]);
    LINFO("A");
    iter = CINcontourRun[s].CONTgetCurrentIter();
    LINFO("iter (next) %d",iter);
    CINIresults[s][iter] = CINcontourRun[s].CONTgetSMI(iter);
    LINFO("C");
    CINresults[iter] += rescale(CINIresults[s][iter] * CINscaleBias[s],
                                CINresults[iter].getWidth(),
                                CINresults[iter].getHeight());
  }
  LINFO("D");
  CINresults[iter] /= CINscales;
  LINFO("E");
  Image<float> resultsRescaled = rescale(CINresults[iter],
                                         CINorignalImageSizeX,
                                         CINorignalImageSizeY);
    /*
    if(t < 10)
      Raster::Visu(resultsRescaled,
                   sformat("results.00%d.out.pgm",t));
    else if(t < 100)
      Raster::Visu(resultsRescaled,
                   sformat("results.0%d.out.pgm",t));
    else
      Raster::Visu(resultsRescaled,
                   sformat("results.%d.out.pgm",t));
    */
  LINFO("F");
  Raster::VisuFloat(resultsRescaled, FLOAT_NORM_0_255,
                   sformat("results.%06d.out.pgm",CINframe));

};

#undef CINNIC2_DEC
#undef CINNIC2_CLASS

// explicit instantiations:
template class CINNIC2<(unsigned short)12, (unsigned short)3,
                       (unsigned short)4, (unsigned short)3, float, int>;

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
