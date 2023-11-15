/*!@file CINNIC/contourRun2.C CINNIC classes - src3 */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourRun2.C $
// $Id: contourRun2.C 6854 2006-07-18 18:24:42Z rjpeters $
//

// ############################################################
// ############################################################
// ##### ---CINNIC2---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#include "CINNIC/contourRun2.H"

#include "Util/Assert.H"
#include "Util/log.H"

#include "Image/ColorOps.H"
#include "Image/ShapeOps.H"

#include <cmath>
#include <cstdlib>

using std::vector;

static float Resistance;  // pulled out of contourRun.H to eliminate warning


//#################################################################
CONTOUR_RUN2_DEC CONTOUR_RUN2_CLASS::contourRun2()
{
  CONTuseFrameSeries = false;
}

//#################################################################
CONTOUR_RUN2_DEC CONTOUR_RUN2_CLASS::~contourRun2()
{
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTtoggleFrameSeries(bool toggle)
{
  CONTuseFrameSeries = toggle;
}

//#################################################################
CONTOUR_RUN2_DEC Image<FLOAT> CONTOUR_RUN2_CLASS::CONTgetSMI(const INT iter)
{
  return CONTsalMapIterations[iter];
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTcopyCombinedSalMap(const std::vector< Image<FLOAT> > &CSM)
{
  CONTcombinedSalMap = CSM;
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTsetConfig(readConfig &config)
{
  CONTtimestep          = config.getItemValueF("timestep");
  CONTmaxEnergy         = config.getItemValueF("maxEnergy");
  Resistance            = config.getItemValueF("Resistance");
  CONTimageSaveTo       = config.getItemValueC("imageSaveTo");
  CONTlogSaveTo         = config.getItemValueC("logSaveTo");
  CONTupperLimit        = config.getItemValueF("upperLimit");
  CONTimageOutDir       = config.getItemValueC("imageOutDir");
  CONTgroupBottom       = config.getItemValueF("groupBottom");
  CONTsupressionAdd     = config.getItemValueF("supressionAdd");
  CONTsupressionSub     = config.getItemValueF("supressionSub");
  CONTadaptType         = (INT)config.getItemValueF("adaptType");
  CONTadaptNeuronThresh = config.getItemValueF("adaptNeuronThresh");
  CONTadaptNeuronMax    = config.getItemValueF("adaptNeuronMax");
  CONTexcMult           = config.getItemValueF("excMult");
  CONTleak              = config.getItemValueF("leak");
  CONTorThresh          = config.getItemValueF("orThresh");
  CONTinitialGroupVal   = config.getItemValueF("initialGroupVal");
  CONTfastPlast         = config.getItemValueF("fastPlast");
  CONTdoFastPlast       = (INT)config.getItemValueF("doFastPlast");
  CONTdoGroupSupression = (INT)config.getItemValueF("doGroupSupression");
  CONTdoPassThrough     = (INT)config.getItemValueF("doPassThrough");
  CONTlastIterOnly      = (INT)config.getItemValueF("lastIterOnly");
  CONTdoTableOnly       = (INT)config.getItemValueF("doTableOnly");
  CONTpassThroughGain   = config.getItemValueF("passThroughGain");
  CONTplastDecay        = config.getItemValueF("plastDecay");
}

//#################################################################
CONTOUR_RUN2_DEC
FLOAT CONTOUR_RUN2_CLASS::CONTsigmoid(const FLOAT beta, const FLOAT v) const
{
  return (1.0f / (1.0f + pow(2.71828f, (-2.0f * (beta * v)))));
}
//#################################################################
CONTOUR_RUN2_DEC
FLOAT CONTOUR_RUN2_CLASS::CONTsigmoid2(const FLOAT beta, const FLOAT v) const
{
  return(1.0f / (1.0f + pow(2.71828f, (-1.0f * (beta+v) ))));
}
//#################################################################
CONTOUR_RUN2_DEC
FLOAT CONTOUR_RUN2_CLASS::CONTpreSigmoid(const FLOAT v, const FLOAT thresh,
                                         const FLOAT beta) const
{
  if(v >= thresh) //optimization (?)
  {
    return (thresh-1);
  }
  else
  {
    const FLOAT sig = CONTsigmoid(beta,(((v/thresh)*(2*(1/beta))-(1/beta))));
    return (sig*thresh);
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTsetImageSize(const INT X, const INT Y)
{
  ASSERT((X > 0) && (Y > 0));
  CONTsetImageSizeY = Y;
  CONTsetImageSizeX = X;
}

//#################################################################
/*! find the max energy per time slice, timestep is 1/x seconds where
  x = timestep
 */
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTderiveEnergy()
{
  CONTenergy = CONTmaxEnergy/CONTtimestep;
}

//#################################################################
/*! this method will reset the neural matrix. Must be called before run image
 */
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTresetMatrix()
{
  // resize static neuron matrix

  std::vector<            staticContourNeuronProp<FLOAT,INT> >   smat1;
  std::vector<std::vector<staticContourNeuronProp<FLOAT,INT> > > smat2;

  staticContourNeuronProp<FLOAT,INT> CONTsprop;
  smat1.resize(            CONTsetImageSizeY       ,CONTsprop);
  smat2.resize(            CONTsetImageSizeX       ,smat1);
  CONTstaticNeuronMatrix.resize(CONTorientations   ,smat2);

  INT ID = 0;

  for(unsigned short i = 0; i < CONTorientations; i++)
  {
    for(INT j = 0; j < CONTsetImageSizeX; j++)
    {
      for(INT k = 0; k < CONTsetImageSizeY; k++)
      {
        CONTstaticNeuronMatrix[i][j][k].sCNP_setID(ID);
        ID++;
      }
    }
  }

  // resize dynamic neuron matrix

  std::vector<                        ContourNeuronProp2<FLOAT,INT> >     mat1;
  std::vector<std::vector<            ContourNeuronProp2<FLOAT,INT> > >   mat2;
  std::vector<std::vector<std::vector<ContourNeuronProp2<FLOAT,INT> > > > mat3;

  ContourNeuronProp2<FLOAT,INT>       CONTprop;
  mat1.resize(            CONTsetImageSizeY ,CONTprop);
  mat2.resize(            CONTsetImageSizeX ,mat1);
  mat3.resize(            CONTorientations  ,mat2);
  CONTneuronMatrix.resize(CONTiterations+2  ,mat3);


  for(unsigned short n = 0; n < CONTiterations+2; n++)
  {
    for(unsigned short i = 0; i < CONTorientations; i++)
    {
      for(INT j = 0; j < CONTsetImageSizeX; j++)
      {
        for(INT k = 0; k < CONTsetImageSizeY; k++)
        {
          CONTneuronMatrix[n][i][j][k].CNP_resetCharge();
          CONTneuronMatrix[n][i][j][k].CNP_linkToStaticMap(
                                       &CONTstaticNeuronMatrix[i][j][k]);
        }
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTresetCharge(const INT iter)
{
  for(unsigned short i = 0; i < CONTorientations; i++)
  {
    for(INT j = 0; j < CONTsetImageSizeX; j++)
    {
      for(INT k = 0; k < CONTsetImageSizeY; k++)
      {
        CONTneuronMatrix[iter][i][j][k].CNP_resetCharge();
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTpreImage(const std::vector< Image<FLOAT> > &imageMap,
                                      const ContourNeuronCreate<FLOAT> &N)
{
  CONTiterCounter = 0;
  CONTderiveEnergy();

  CONTsalMap.resize(CONTsetImageSizeX,CONTsetImageSizeY,true);
  CONTsalMapIterations.resize(CONTiterations,CONTsalMap);

  CONTgroupMap.resize(CONTiterations,CONTsalMap);
  CONTimageOpt.resize(CONTorientations,imageMap[1]);

  //move through the entire image  ImageMap ImageOpt;
  CONTsetImageOpt(imageMap,true);
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTsetImageOpt(const std::vector< Image<FLOAT> > &imageMap,
                                         bool resize)
{
  for(unsigned short a = 0; a < CONTorientations; a++)
  {
    if(resize == true)
      CONTimageOpt[a].resize(imageMap[a].getWidth(),imageMap[a].getHeight(),ZEROS);
    for(INT i = 0; i < CONTsetImageSizeX; i++)
    {
      for(INT j = 0; j < CONTsetImageSizeY; j++)
      {
        CONTimageOpt[a].setVal(i,j,(imageMap[a].getVal(i,j)*CONTenergy));
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTcalcSalMap(const std::vector< Image<FLOAT> > &imageMap,
                                        const INT iter)
{
  //First check potentials for each column at this iteration,
  //reset if nessessary
  //First check potentials for each column at this iteration,
  //reset if nessessary
  for(INT a = 0; a < CONTorientations; a++)
  {
    for(INT i = 0; i < CONTsetImageSizeX; i++)
    {
      for(INT j = 0; j < CONTsetImageSizeY; j++)
      {
        //Add up all charges in this column
        //if charge is negative then make zero
        if(CONTneuronMatrix[iter][a][i][j].CNP_getCharge() < 0)
        {
          CONTneuronMatrix[iter][a][i][j].CNP_resetCharge();
        }
        // add this iteration plus all others combined here
        // i.e. here lies the integrator for each pixel!
        const FLOAT hold = CONTsalMap.getVal(i,j) +
          CONTneuronMatrix[iter][a][i][j].CNP_getCharge();

        CONTsalMap.setVal(i,j,hold);
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTprocessSalMap(
                                     const std::vector< Image<FLOAT> > &imageMap,
                                     const INT iter)
{
  //CONTsalMapIterations[iter].resize(CONTsetImageSizeX,CONTsetImageSizeY,true);
  //CONTgroupMap[iter].resize(CONTsetImageSizeX,CONTsetImageSizeY,true);
  for(INT i = 0; i < CONTsetImageSizeX; i++)
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++)
    {
      // leak this neuron
      FLOAT hold1 = CONTsalMap.getVal(i,j) - CONTleak;
      // bottom this neuron to 0 charge if negative
      if(hold1 < 0){hold1 = 0;}
      // Set the sal map to this value
      CONTsalMap.setVal(i,j,hold1);

      // Compute sigmoid of column this iteration
      const FLOAT hold2 =
        CONTpreSigmoid(CONTsalMap.getVal(i,j),CONTupperLimit);
      // set output (for movie) to a value from 0 to 256
      CONTgroupMap[iter].setVal(i,j,hold2);
      // set value into this iteration normalized
      if((i > 0) && (j > 0))
      {
        CONTsalMapIterations[iter].setVal((i-1),(j-1)
                                          ,((hold2/CONTupperLimit)*255));
      }
    }
  }
}

/*
   OUTER LOOP/GROUP SUPRESSION COMPUTATION
   here we execute outer portions of the 6 layer loop in CINNIC
   for psuedo-convolution
   We also compute the group supression based upon change in groups
   overall total activity change (derivative)
*/
//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTcalcGroups(const std::vector< Image<FLOAT> > &imageMap,
                                        const INT iter, const INT lastIter,
                                        const bool init)
{
  // calculate group DELTA if group adaptation selected
  for(INT i = 0; i < CONTgroups; i++)
  {
    CONTgroupDelta[i] = 0;
  }

  if((init == false) && (CONTadaptType == 1))
  {
    for(INT i = 0; i < CONTsetImageSizeX; i++)
    {
      for(INT j = 0; j < CONTsetImageSizeY; j++)
      {
        //what group is this column in?
        const INT hold = (INT)CONTgroup.getVal(i,j);
        //what was the last iteration
        // find delta value for sig. on column, add to group
        // That is, what has this group changed by (derivative)
        CONTgroupDelta[hold] += CONTgroupMap[iter].getVal(i,j) -
                                CONTgroupMap[lastIter].getVal(i,j);
      }
    }
  }

  // modify supression values using groups
  if(CONTadaptType == 1)
  {
    for(INT g = 0; g < CONTgroups; g++)
    {
      // if values are too big then add suppression
      // this is done per pixel group
      if(CONTgroupDelta[g] > CONTgroupTop)
      {
        CONTgroupMod[g] += CONTsupressionAdd *
                           (CONTgroupDelta[g] - CONTgroupTop);
        CONTgroupMod2[g] = 1 / CONTgroupMod[g];
      }
      // if values are too small then remove supression
      if(CONTgroupDelta[g] < CONTgroupBottom)
      {
        CONTgroupMod[g] -= CONTsupressionSub *
          fabs(CONTgroupBottom - CONTgroupDelta[g]);
      }
      // bound group supression
      if(CONTgroupMod[g] > CONTmaxGroupSupress)
        CONTgroupMod[g] = CONTmaxGroupSupress;
      else if(CONTgroupMod[g] < CONTminGroupSupress)
        CONTgroupMod[g] = CONTminGroupSupress;
    }
  }
  CONTiterCounter = 0;
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTsetGroupPointers(
                                     const std::vector< Image<FLOAT> > &imageMap,
                                     const INT a, const INT iter)
{
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if(imageMap[a].getVal(i,j) > CONTsmallNumber) //optimization
      {
        const INT thisGroup = (INT)CONTgroup.getVal(i,j);
        CONTstaticNeuronMatrix[a][i][j].
          sCNP_setGroupMod(&CONTgroupMod[thisGroup]);
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC inline
void CONTOUR_RUN2_CLASS::CONTfindFastPlasticity(
                                     const std::vector< Image<FLOAT> > &imageMap,
                                     const INT a, const INT iter)
{
  const FLOAT min = CONTminFastPlasticity;
  const FLOAT max = CONTmaxFastPlasticity;
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if(imageMap[a].getVal(i,j) > CONTsmallNumber) //optimization
      {
        // FAST PLASTICITY
        //MUST KILL OTHER NEURONS!!!! - fast placicity
        //sigmoid this at some point ?
        FLOAT plast =
          CONTneuronMatrix[iter][a][i][j].CNP_getCharge()*CONTfastPlast -
          CONTplastDecay;
        if(plast < min)
        {
           plast = min;
        }
        // upper bound on fast plasticity
        else if(plast > max)
        {
          plast = max;
        }
        CONTneuronMatrix[iter][a][i][j].CNP_setFastPlast(plast);
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC inline
void CONTOUR_RUN2_CLASS::CONTfindPassThroughGain(
                                     const std::vector< Image<FLOAT> > &imageMap,
                                     const INT a, const INT iter,
                                     const INT nextIter)
{
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      //LINFO("%d,%d",i,j);
      if(imageMap[a].getVal(i,j) > CONTsmallNumber) //optimization
      {
        // PASS THROUGH GAIN
        //LINFO("OK");
        const INT thisGroup = (INT)CONTgroup.getVal(i,j);
        //LINFO("AA");
        //LINFO("This Group %d",thisGroup);
        const FLOAT passThrough = imageMap[a].getVal(i,j) *
                                  (CONTpassThroughGain /
                                  ((CONTgroupMod[thisGroup] * 5) - 4));
        //LINFO("BB");
        // set pass through gain for next iter
        CONTneuronMatrix[nextIter][a][i][j].CNP_chargeSimple(passThrough);
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTrunImageSigmoid(
                             const std::vector< Image<FLOAT> > &imageMap,
                             const ContourNeuronCreate<FLOAT> &N,
                             const INT iter, const INT nextIter,
                             const INT lastIter, const bool init)
{
  LINFO("lastIter %d, iter %d, nextIter %d",lastIter,iter,nextIter);
  Timer tim;
  tim.reset();
  int t1,t2,t3;
  int t0 = tim.get();  // to measure display time
  LINFO("Calculating Salmap");
  CONTcalcSalMap(imageMap,iter);
  t1 = tim.get();
  t2 = t1 - t0; t3 = t2;
  LINFO("TIME: %d ms Slice: %d ms",t2,t3);
  LINFO("Processing Salmap");
  CONTprocessSalMap(imageMap,iter);
  t1 = tim.get();
  t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
  LINFO("TIME: %d ms Slice: %d ms",t2,t3);
  if(CONTdoGroupSupression == 1)
  {
    LINFO("Calculating groups");
    CONTcalcGroups(imageMap,iter,lastIter,init);
    t1 = tim.get();
    t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
    LINFO("TIME: %d ms Slice: %d ms",t2,t3);
  }
  LINFO("Running Pseudo Convolution");
  CONTiterateConvolve(imageMap,N,-1,iter,nextIter,init);
  t1 = tim.get(); t3 = t2; t2 = t1 - t0; t3 = t2 - t3;
  LINFO("TIME: %d ms Slice: %d ms",t2,t3);
}


//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTiterateConvolve(
                                     const std::vector< Image<FLOAT> > &imageMap,
                                     const ContourNeuronCreate<FLOAT> &N,
                                     const INT node, const INT iter,
                                     const INT nextIter, const bool init)
{
  //RUN hypercolumn get charges for i put charges for i+1
  for(unsigned short a = 0; a < CONTorientations; a++) //this neuron's angle
  {
    Raster::VisuFloat(CONTimageOpt[a], FLOAT_NORM_0_255,
                      sformat("input1.%06d.%d.%d.out.pgm",CONTcurrentFrame,a,
                              CONTsetImageSizeX));
    Raster::VisuFloat(imageMap[a], FLOAT_NORM_0_255,
                      sformat("input2.%06d.%d.%d.out.pgm",CONTcurrentFrame,a,
                              CONTsetImageSizeX));
    //LINFO("A");
    if(init == true)
      CONTsetGroupPointers(imageMap,a,iter);
    //LINFO("B");
    if(CONTdoFastPlast == 1)
      CONTfindFastPlasticity(imageMap,a,iter);
    //LINFO("C");
    if(CONTdoPassThrough == 1)
      CONTfindPassThroughGain(imageMap,a,iter,nextIter);
    if(init == true)
    {
      //other neuron's angle
      for(unsigned short b = 0; b < CONTorientations; b++)
      {
        //CONTconvolveSimpleOld(imageMap,N,a,b,node,iter);
        // LINFO("D");
        CONTconvolveSimpleInit(imageMap,N,a,b,node,iter,nextIter);
      }
    }
    else
    {
      // LINFO("D2");
      if(CONTuseFrameSeries == false)
        CONTconvolveSimple(imageMap,N,a,node,iter,nextIter);
      else
        CONTconvolveSimpleFrames(imageMap,N,a,node,iter,nextIter);
    }
  }
}

/*

---------Psuedo-Convolution Core for CINNIC. Current version-----------

This is the meat of CINNIC. Here the hyper column is looped over
   for all neurons and interactions. While the main loop is only
   6 nested levels deep, multiple optimizations are added to test
   if a neuron should be used. For instance if it is zero skip it

*/

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTconvolveSimpleInit(
                                    const std::vector< Image<FLOAT> > &imageMap,
                                    const ContourNeuronCreate<FLOAT> &N,
                                    const INT a, const INT b, const INT node,
                                    const INT iter, const INT nextIter)
{
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if((imageMap[a].getVal(i,j) > CONTsmallNumber) ||
         (CONTuseFrameSeries == true))   //optimization
      {
        const INT thisGroup = (INT)CONTgroup.getVal(i,j);
        //Other Neuron's position X
        for(INT k = 0; k <= CONTkernelSize; k++)
        {
          //Current position plus its field - center
          const INT InspX = i + (k - (INT)XCenter);
          if(InspX >= 0)
          {
            if(InspX < CONTsetImageSizeX)
            {
              //Other Neuron's position Y
              for(INT l = 0; l <= CONTkernelSize; l++)
              {
                const INT InspY = j - (l-(INT)YCenter);
                if(InspY >= 0) //stay inside of image
                {
                  if(InspY < CONTsetImageSizeY)
                  {
                    //check that this element != 0
                    if(N.FourDNeuralMap[a][b][k][l].zero)
                    {
                      // optimization
                      if(imageMap[b].getVal(InspX,InspY) > CONTorThresh )
                      {
                        FLOAT hold;
                        bool polarity;
                        // this is done this way to optimize the code
                        // for multiple
                        // iteration, by storing some of the first iterations
                        // results in an array

                        // ## it is important to note that ImageOpt
                        // = Image*energy, thus it's an optimization
                        // if this is a negative number ,
                        // then supression may be modified

                        // FORMULA HERE
                        // orient filered image1 * orient filtered image2
                        // * their excitation
                        const FLOAT weight =
                          ((CONTimageOpt[a].getVal(i,j) *
                            imageMap[b].getVal(InspX,InspY)) *
                           N.FourDNeuralMap[a][b][k][l].angABD);
                        //CONTstoreVal[CONTiterCounter] =
                        // apply group supression if < 0
                        if(N.FourDNeuralMap[a][b][k][l].angABD < 0)
                        {
                          polarity = true;
                          hold =
                            CONTneuronMatrix[iter][a][i][j].
                            CNP_computeSupress(weight,CONTgroupMod[thisGroup]);
                          /* old
                          hold =
                            CONTgroupMod[thisGroup] *
                            mod * CONTstoreVal[CONTiterCounter];
                          */
                        }
                        else // value is > 0, no group supression
                        {
                          polarity = false;
                          hold =
                            CONTneuronMatrix[iter][a][i][j].
                            CNP_computeExcite(weight,CONTgroupMod[thisGroup]);
                        }
                        // set energy prior to sal map
                        CONTneuronMatrix[nextIter][b][InspX][InspY].
                          CNP_chargeSimple(hold);
                        //! set connection between these neurons as active
                        if(CONTuseFrameSeries == false)
                        {
                          CONTstaticNeuronMatrix[a][i][j].
                            sCNP_insertStoreList((unsigned char)b,
                                                 (unsigned char)InspX,
                                                 (unsigned char)InspY,
                                                 polarity,weight);
                        }
                        else
                        {
                          CONTstaticNeuronMatrix[a][i][j].
                            sCNP_insertStoreList((unsigned char)b,
                                                 (unsigned char)InspX,
                                                 (unsigned char)InspY,
                                                 polarity,
                                          N.FourDNeuralMap[a][b][k][l].angABD);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTconvolveSimpleOld(
                                    const std::vector< Image<FLOAT> > &imageMap,
                                    const ContourNeuronCreate<FLOAT> &N,
                                    const INT a, const INT b, const INT node,
                                    const INT iter, const INT nextIter)
{
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if(imageMap[a].getVal(i,j) > CONTsmallNumber) //optimization
      {
        const INT thisGroup = (INT)CONTgroup.getVal(i,j);
        float mod;
        mod = CONTneuronMatrix[iter][a][i][j].CNP_getCharge()*CONTfastPlast;
        if(mod < 1){mod = 1;}
        if(mod > 5){mod = 5;}
        CONTneuronMatrix[iter][a][i][j].CNP_setFastPlast(mod);
        float crap = imageMap[a].getVal(i,j)*
          (CONTpassThroughGain/((CONTgroupMod[thisGroup]*5)-4));
        CONTneuronMatrix[nextIter][a][i][j].CNP_chargeSimple(crap);
        //Other Neuron's position X
        for(INT k = 0; k <= CONTkernelSize; k++)
        {
          //Current position plus its field - center
          const INT InspX = i + (k - (INT)XCenter);
          if(InspX >= 0)
          {
            if(InspX < CONTsetImageSizeX)
            {
              //Other Neuron's position Y
              for(INT l = 0; l <= CONTkernelSize; l++)
              {
                const INT InspY = j - (l-(INT)YCenter);
                if(InspY >= 0) //stay inside of image
                {
                  if(InspY < CONTsetImageSizeY)
                  {
                    //check that this element != 0
                    if(N.FourDNeuralMap[a][b][k][l].zero)
                    {
                      // optimization
                      if(imageMap[b].getVal(InspX,InspY) > CONTorThresh )
                      {
                        FLOAT hold;
                        // this is done this way to optimize the code
                        // for multiple
                        // iteration, by storing some of the first iterations
                        // results in an array

                        // ## it is important to note that ImageOpt
                        // = Image*energy, thus it's an optimization
                        // if this is a negative number ,
                        // then supression may be modified

                        // FORMULA HERE
                        // orient filered image1 * orient filtered image2
                        // * their excitation
                        const FLOAT weight =
                          ((CONTimageOpt[a].getVal(i,j) *
                            imageMap[b].getVal(InspX,InspY)) *
                           N.FourDNeuralMap[a][b][k][l].angABD);
                        //CONTstoreVal[CONTiterCounter] =
                        // apply group supression if < 0
                        if(N.FourDNeuralMap[a][b][k][l].angABD < 0)
                        {
                          hold =
                            CONTneuronMatrix[iter][a][i][j].
                            CNP_computeSupress(weight,CONTgroupMod[thisGroup]);
                          /* old
                          hold =
                            CONTgroupMod[thisGroup] *
                            mod * CONTstoreVal[CONTiterCounter];
                          */
                        }
                        else // value is > 0, no group supression
                        {
                          hold =
                            CONTneuronMatrix[iter][a][i][j].
                            CNP_computeExcite(weight,CONTgroupMod[thisGroup]);
                        }
                        // set energy prior to sal map
                        CONTneuronMatrix[nextIter][b][InspX][InspY].
                          CNP_chargeSimple(hold);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTconvolveSimple(
                                    const std::vector< Image<FLOAT> > &imageMap,
                                    const ContourNeuronCreate<FLOAT> &N,
                                    const INT a, const INT node,
                                    const INT iter, const INT nextIter)
{
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if(imageMap[a].getVal(i,j) > CONTsmallNumber) //optimization
      {
        const INT thisGroup = (INT)CONTgroup.getVal(i,j);
        for(unsigned int n = 0;
            n < CONTstaticNeuronMatrix[a][i][j].sCNP_getActiveNeuronCount();
            n++)
        {
          FLOAT hold;
          if(CONTstaticNeuronMatrix[a][i][j].sCNP_getOtherNeuronPol(n))
          {
            hold =
              CONTneuronMatrix[iter][a][i][j].
              CNP_computeSupress(CONTstaticNeuronMatrix[a][i][j].
                                 sCNP_getWeightStoreVal(n),
                                 CONTgroupMod[thisGroup]);
          }
          else // value is > 0, no group supression
          {
            hold =
              CONTneuronMatrix[iter][a][i][j].
              CNP_computeExcite(CONTstaticNeuronMatrix[a][i][j].
                                sCNP_getWeightStoreVal(n),
                                CONTgroupMod[thisGroup]);
          }
          // set energy prior to sal map
          CONTneuronMatrix[nextIter]
            [CONTstaticNeuronMatrix[a][i][j].
             sCNP_getOtherNeuronAlpha(n)]
            [CONTstaticNeuronMatrix[a][i][j].
             sCNP_getOtherNeuron_i(n)]
            [CONTstaticNeuronMatrix[a][i][j].
             sCNP_getOtherNeuron_j(n)].
            CNP_chargeSimple(hold);
        }
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTconvolveSimpleFrames(
                                    const std::vector< Image<FLOAT> > &imageMap,
                                    const ContourNeuronCreate<FLOAT> &N,
                                    const INT a, const INT node,
                                    const INT iter, const INT nextIter)
{

  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if(imageMap[a].getVal(i,j) > CONTsmallNumber) //optimization
      {
        const INT thisGroup = (INT)CONTgroup.getVal(i,j);
        for(unsigned int n = 0;
            n < CONTstaticNeuronMatrix[a][i][j].sCNP_getActiveNeuronCount();
            n++)
        {
          const unsigned char other_i  =
            CONTstaticNeuronMatrix[a][i][j].sCNP_getOtherNeuron_i(n);
          const unsigned char other_j  =
            CONTstaticNeuronMatrix[a][i][j].sCNP_getOtherNeuron_j(n);
          const unsigned char other_a  =
            CONTstaticNeuronMatrix[a][i][j].sCNP_getOtherNeuronAlpha(n);

          if(imageMap[other_a].getVal(other_i,other_j) > CONTsmallNumber)
          {
            const FLOAT neuralMapWeight =
              CONTstaticNeuronMatrix[a][i][j].sCNP_getWeightStoreVal(n);

            const FLOAT weight = ((CONTimageOpt[a].getVal(i,j) *
                                   imageMap[other_a].getVal(other_i,other_j)) *
                                  neuralMapWeight);

            FLOAT hold;
            if(CONTstaticNeuronMatrix[a][i][j].sCNP_getOtherNeuronPol(n))
            {
              hold =
                CONTneuronMatrix[iter][a][i][j].
                CNP_computeSupress(weight,CONTgroupMod[thisGroup]);
            }
            else // value is > 0, no group supression
            {
              hold =
                CONTneuronMatrix[iter][a][i][j].
                CNP_computeExcite(weight,CONTgroupMod[thisGroup]);
            }
            // set energy prior to sal map
            CONTneuronMatrix[nextIter][other_a][other_i][other_j].
              CNP_chargeSimple(hold);
          }
        }
      }
    }
  }
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTcontourRunMain(
                                        const std::vector< Image<FLOAT> > &imageMap,
                                        const ContourNeuronCreate<FLOAT> &N,
                                        readConfig &config,
                                        const Image<FLOAT> &group,
                                        const INT groups,
                                        const INT iter,
                                        const FLOAT groupTop)
{
  bool init = false;
  if(iter == 0)
  {
    CONTgroupTop = groupTop;
    CONTgroup    = group;
    CONTgroups   = groups;
    CONTsetConfig(config);
    CONTsetImageSize(imageMap[1].getWidth(),imageMap[1].getHeight());
    CONTresetMatrix();
    CONTpreImage(imageMap,N);
    // groups can be set to either 1 or 0 depending on whether or not you
    // want supression to happen only when a group reaches threshold
    // excitation or not.
    CONTgroupMod.resize(groups,CONTinitialGroupVal);
    CONTgroupMod2.resize(groups,CONTinitialGroupVal);
    CONTgroupDelta.resize(groups,0.0F);
    init = true;
  }
  CONTiterCounter = 0;
  INT lastIter = iter-1;
  INT nextIter = iter+1;
  CONTcurrentFrame = CONTcurrentIter;
  CONTrunImageSigmoid(imageMap,N,iter,nextIter,lastIter,init);
  CONToutputFastPlasticity(iter);
  CONToutputGroupSupression(iter);
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONTcontourRunFrames(
                                        const std::vector< Image<FLOAT> > &imageMap,
                                        const ContourNeuronCreate<FLOAT> &N,
                                        readConfig &config,
                                        const Image<FLOAT> &group,
                                        const INT groups,
                                        const INT frame,
                                        const FLOAT groupTop)
{
  bool init = false;
  CONTcurrentFrame = frame;
  if(frame == 1)
  {
    CONTgroupTop    = groupTop;
    CONTgroup       = group;
    CONTgroups      = groups;
    CONTcurrentIter = 0;
    CONTsetConfig(config);
    CONTsetImageSize(imageMap[1].getWidth(),imageMap[1].getHeight());
    CONTresetMatrix();
    CONTpreImage(imageMap,N);
    // groups can be set to either 1 or 0 depending on whether or not you
    // want supression to happen only when a group reaches threshold
    // excitation or not.
    CONTgroupMod.resize(groups,CONTinitialGroupVal);
    CONTgroupMod2.resize(groups,CONTinitialGroupVal);
    CONTgroupDelta.resize(groups,0.0F);
    init = true;
  }
  CONTiterCounter = 0;
  INT lastIter;
  INT nextIter;

  LINFO("current %d iter %d",CONTcurrentIter,CONTiterations);
  if(CONTcurrentIter == (CONTiterations - 1))
  {
    lastIter = CONTiterations - 2;
    nextIter = 0;
    LINFO("1");
  }
  else if(CONTcurrentIter == 0)
  {
    lastIter = CONTiterations - 1;
    nextIter = 1;
    LINFO("2");
  }
  else
  {
    lastIter = CONTcurrentIter - 1;
    nextIter = CONTcurrentIter + 1;
    LINFO("3");
  }
  LINFO("%d iter, %d nextIter, %d lastIter",CONTcurrentIter,nextIter,lastIter);
  CONTresetCharge(nextIter);
  CONTsetImageOpt(imageMap,false);
  CONTrunImageSigmoid(imageMap,N,CONTcurrentIter,nextIter,lastIter,init);
  CONToutputFastPlasticity(CONTcurrentIter);
  CONToutputGroupSupression(CONTcurrentIter);

  // since all data is stored in the next iter, that's where the calling
  // methods should look for results
  CONTstoreCurrentIter = nextIter;

  if(CONTcurrentIter == (CONTiterations - 1))
    CONTcurrentIter = 0;
  else
    CONTcurrentIter++;
}

//#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONToutputFastPlasticity(INT iter)
{
  Image<float> output;
  output.resize(CONTsetImageSizeX,CONTsetImageSizeY);
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      output.setVal(i,j,0.0F);
    }
  }

  // sum image values for each hyper column
  for(INT a = 0; a < CONTorientations; a++)
  {
    for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
    {
      for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
      {
        output.setVal(i,j,CONTneuronMatrix[iter][a][i][j].CNP_getFastPlast() +
                      output.getVal(i,j));
      }
    }
  }

  // find max value to normalize image
  float maxVal = 0;
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if(output.getVal(i,j) > maxVal){ maxVal = output.getVal(i,j);}
    }
  }

  // create image with blue/green normalized by max value possible and
  // red normalized by max value observed. This normalizes the output
  // but gives us an idea of scale. Scale is higher if output is more white.
  Image<PixRGB<float> > RGoutput;
  RGoutput.resize(CONTsetImageSizeX,CONTsetImageSizeY);
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      const FLOAT val1 = (output.getVal(i,j)/
                          (CONTmaxFastPlasticity*CONTorientations)) * 255.0F;
      const FLOAT val2 = (output.getVal(i,j)/maxVal) * 255.0F;
      const PixRGB<float> pix(val2,val1,val1);
      RGoutput.setVal(i,j,pix);
    }
  }

  RGoutput = rescale(RGoutput,CONTsetImageSizeX*4,CONTsetImageSizeY*4);

  Raster::VisuRGB(RGoutput, sformat("fastPlast.%d.%06d.out.ppm",
                                    CONTsetImageSizeX,CONTcurrentFrame));
}

  //#################################################################
CONTOUR_RUN2_DEC
void CONTOUR_RUN2_CLASS::CONToutputGroupSupression(INT iter)
{
  Image<float> output;
  output.resize(CONTsetImageSizeX,CONTsetImageSizeY);
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      output.setVal(i,j,0.0F);
    }
  }

  for(INT a = 0; a < CONTorientations; a++)
  {
    for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
    {
      for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
      {
        const INT thisGroup = (INT)CONTgroup.getVal(i,j);
        output.setVal(i,j,CONTgroupMod[thisGroup] + output.getVal(i,j));
      }
    }
  }

  // find max value to normalize image
  float maxVal = 0;
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      if(output.getVal(i,j) > maxVal){ maxVal = output.getVal(i,j);}
    }
  }

  // create image with blue/green normalized by max value possible and
  // red normalized by max value observed. This normalizes the output
  // but gives us an idea of scale. Scale is higher if output is more white.
  Image<PixRGB<float> > RGoutput;
  RGoutput.resize(CONTsetImageSizeX,CONTsetImageSizeY);
  for(INT i = 0; i < CONTsetImageSizeX; i++) //This Neuron's position X
  {
    for(INT j = 0; j < CONTsetImageSizeY; j++) //This Neuron's position Y
    {
      const FLOAT val1 = (output.getVal(i,j)/
                          (CONTmaxGroupSupress*CONTorientations)) * 255.0F;
      const FLOAT val2 = (output.getVal(i,j)/maxVal) * 255.0F;
      const PixRGB<float> pix(val2,val1,val1);
      RGoutput.setVal(i,j,pix);
    }
  }

  RGoutput = rescale(RGoutput,CONTsetImageSizeX*4,CONTsetImageSizeY*4);

  Raster::VisuRGB(RGoutput, sformat("groupSup.%d.%06d.out.ppm",
                                    CONTsetImageSizeX,CONTcurrentFrame));
}

// ######################################################################
CONTOUR_RUN2_DEC
INT CONTOUR_RUN2_CLASS::CONTgetCurrentIter()
{
  return CONTstoreCurrentIter;
}

#undef CONTOUR_RUN2_DEC
#undef CONTOUR_RUN2_CLASS


// explicit instantiations:
#define CR2INST contourRun2<(unsigned short)12, (unsigned short)3, \
    (unsigned short)4, (unsigned short)3, float, int>

template class CR2INST;
template <> const float CR2INST::CONTmaxFastPlasticity     = 5.0F;
template <> const float CR2INST::CONTminFastPlasticity     = 1.0F;
template <> const float CR2INST::CONTmaxGroupSupress       = 10.0F;
template <> const float CR2INST::CONTminGroupSupress       = 1.0F;
template <> const float CR2INST::CONTsmallNumber           = 0.001F;



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
