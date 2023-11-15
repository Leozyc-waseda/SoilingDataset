/*!@file CINNIC/contourRun.C CINNIC classes - src3 */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourRun.C $
// $Id: contourRun.C 6191 2006-02-01 23:56:12Z rjpeters $
//

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#include "CINNIC/contourRun.H"

#include "Util/Assert.H"
#include "Image/ColorOps.H"
#include "Image/ShapeOps.H"
#include "Util/log.H"

#include <cmath>
#include <cstdlib>
#include <fstream>

static float Resistance;  // pulled out of contourRun.H to eliminate warning


//#################################################################
contourRun::contourRun()
{
}

//#################################################################
contourRun::~contourRun()
{
}

//#################################################################
Image<float> contourRun::getSMI(int iter)
{
  return SMI[iter];
}

//#################################################################
void contourRun::copyCombinedSalMap(std::vector< Image<float> > &CSM)
{
  combinedSalMap = CSM;
}

//#################################################################
void contourRun::setConfig(readConfig &config)
{
  iterations = (int)config.getItemValueF("iterations");
  timestep = config.getItemValueF("timestep");
  maxEnergy = config.getItemValueF("maxEnergy");
  BaseThreshold = config.getItemValueF("BaseThreshold");
  Resistance = config.getItemValueF("Resistance");
  imageSaveTo = config.getItemValueC("imageOutDir");
  logSaveTo = config.getItemValueC("logOutDir");
  dumpSwitchPos = (int)config.getItemValueF("dumpSwitchPos");
  upperLimit = config.getItemValueF("upperLimit");
  saveto = config.getItemValueC("imageOutDir");
  iTrans = config.getItemValueF("overlayDumpValue");
  GroupBottom = config.getItemValueF("GroupBottom");
  supressionAdd = config.getItemValueF("supressionAdd");
  supressionSub = config.getItemValueF("supressionSub");
  cascadeType = (int)config.getItemValueF("cascadeType");
  adaptType = (int)config.getItemValueF("adaptType");
  adaptNeuronThresh = config.getItemValueF("adaptNeuronThresh");
  adaptNeuronMax = config.getItemValueF("adaptNeuronMax");
  excMult =  config.getItemValueF("excMult");
  leak = config.getItemValueF("leak");
  orThresh = config.getItemValueF("orThresh");
  initialGroupVal = config.getItemValueF("initialGroupVal");
  fastPlast = config.getItemValueF("fastPlast");
  doFastPlast = (int)config.getItemValueF("doFastPlast");
  lastIterOnly = (int)config.getItemValueF("lastIterOnly");
  doTableOnly = (int)config.getItemValueF("doTableOnly");
  passThroughGain = config.getItemValueF("passThroughGain");
  passThroughTaper = config.getItemValueF("passThroughTaper");
}
//#################################################################
void contourRun::setArraySize(long size)
{
  storeVal = new float[size];
}

//#################################################################
float contourRun::sigmoid(float beta, float v)
{
  return (1.0f / (1.0f + pow(2.71828f, (-2.0f * (beta * v)))));
}
//#################################################################
float contourRun::sigmoid2(float beta, float v)
{
  return(1.0f / (1.0f + pow(2.71828f, (-1.0f * (beta+v) ))));
}
//#################################################################
float contourRun::preSigmoid(float v, float thresh, float beta)
{
  if(v >= thresh) //optimization (?)
  {
    return (thresh-1);
  }
  else
  {
    float sig = sigmoid(beta,(((v/thresh)*(2*(1/beta))-(1/beta))));
    if((sig*thresh) > thresh){LINFO("FOO %f",(sig*thresh));}
    return (sig*thresh);
  }
}
//#################################################################
void contourRun::dumpEnergySigmoid(const char* filename, const char* savefile,
                                   readConfig &config, Image<float> image,
                                   int scaleNo, int scaleTot)
{
  ASSERT(scaleNo >= 0);ASSERT(scaleTot > 0);
  Image< byte > Tyte;
  Image<PixRGB<byte> > TPyte;
  Image<float> SMIr;
  std::ofstream propFile("CINNIC.prop",std::ios::out);
  propFile << "scale " << scaleTot << "\n"
           << "iterations " << iterations << "\n"
           << "sizeX " << image.getWidth() << "\n"
           << "sizeY " << image.getHeight() << "\n";
  int sx = image.getWidth(); int sy = image.getHeight();
  propFile << "type potential\n"; //output to propfile
  propFile << "type cascade\n"; //output to propfile
  propFile << "type overlay\n"; //output to propfile
  propFile.close();
  for(int i = 0; i < iterations; i++)
  {
    SMIr.resize(SMI[i].getWidth(),SMI[i].getHeight());
    for(int x = 0; x < SMI[i].getWidth(); x++)
    {
      for(int y = 0; y < SMI[i].getHeight(); y++)
      {
        if((x == 0) || (y == 0))
        {
          SMIr.setVal(x,y,0);
        }
        else
        {
          SMIr.setVal(x,y,SMI[i].getVal(x-1,y-1));
        }
      }
    }

    SMIr = rescale(SMIr, sx,sy);
    Tyte = SMIr;
    int xx = Tyte.getWidth();
    int yy = Tyte.getHeight();
    // do not write images other then the last one if value is set
    if(((lastIterOnly == 0) || (i == (iterations - 1))) && (doTableOnly == 0))
    {
      Raster::WriteRGB(Image<PixRGB<byte> >(Tyte),
                       sformat("%s%s.potential.out.%d.%d.%d.%d.ppm"
                               ,saveto,savefile,scaleNo,i,xx,yy));
      TPyte = cascadeMap[i];
      overlayS = overlayStain(SMIr,image,iTrans,'r');
      TPyte = overlayS;
      Raster::WriteRGB(TPyte,
                       sformat("%s%s.overlay.out.%d.%d.%d.%d.ppm"
                               ,saveto,savefile,scaleNo,i,xx,yy));
    }
    LINFO("CRAP");
  }
}
//#################################################################
void contourRun::setImageSize(int X, int Y)
{
  ASSERT((X > 0) && (Y > 0));
  setImageSizeY = Y;
  setImageSizeX = X;
}
//#################################################################
void contourRun::setIterations(int iter)
{
  ASSERT(iter > 0);
  iterations = iter;
}
//#################################################################
/*! find the max energy per time slice, timestep is 1/x seconds where
  x = timestep
 */
void contourRun::deriveEnergy()
{
  energy = maxEnergy/timestep;
}
//#################################################################
/*! this method will reset the neural matrix. Must be called before run image
 */
void contourRun::resetMatrix()
{
  int it = (int)iterations+2;
  mat1.resize(setImageSizeY,prop);
  mat2.resize(setImageSizeX,mat1);
  mat3.resize(AnglesUsed,mat2);
  NeuronMatrix.resize(it,mat3);
  //LINFO("NeuronMatrix vectors set");
  float foo = BaseThreshold;
  for(int n = 0; n < (int)(iterations+2.0F); n++)
  {
    for(int i = 0; i < AnglesUsed; i++)
    {
      for(int j = 0; j < setImageSizeX; j++)
      {
        for(int k = 0; k < setImageSizeY; k++)
        {
          NeuronMatrix[n][i][j][k].ResetTempCharge();
          NeuronMatrix[n][i][j][k].ResetCharge();
          NeuronMatrix[n][i][j][k].setThreshold(foo,Resistance);
        }
      }
    }
  }
  //LINFO("reset done");
}

//#################################################################
void contourRun::preImage(std::vector< Image<float> > &imageMap,
                          ContourNeuronCreate<float> &N)
{
  iterCounter = 0;
  cascadeChunk = 100;
  deriveEnergy();
  SM.resize(setImageSizeX,setImageSizeY,true);
  SMI.resize((int)iterations,SM);
  cascadeMap.resize((int)iterations,
                    Image<PixRGB<float> >(setImageSizeX, setImageSizeY, ZEROS));
  GroupMap.resize((int)iterations,SM);
  cascade.resize(cascadeChunk);
  cascadeSize.resize(setImageSizeX,setImageSizeY);
  ICH.resize(setImageSizeX,setImageSizeY,true);
  cascadeImage.resize(cascadeChunk,ICH);
  imageOpt.resize(AnglesUsed,imageMap[1]);

  //move through the entire image  ImageMap ImageOpt;
  for(int a = 0; a < AnglesUsed; a++)
  {
    imageOpt[a].resize(imageMap[a].getWidth(),imageMap[a].getHeight(),true);
    for(int i = 0; i < setImageSizeX; i++)
    {
      for(int j = 0; j < setImageSizeY; j++)
      {
        imageOpt[a].setVal(i,j,(imageMap[a].getVal(i,j)*energy));
      }
    }
  }
#if 0
  /* commented this loop out because

     (1) it triggers a bug in g++ 3.4.1, but

     (2) this loop should be unneccessary in any case because in the above
         line:

         cascadeImage.resize(cascadeChunk,ICH);

         cascadeImage gets set to have 'cascadeChunk' number of copies of
         ICH, and ICH was already resize'd to
         (setImageSizeX,setImageSizeY). So... each cascadeImage[i] should
         already have the desired size, so we can skip this loop, and we
         can therefore avoid triggering the bug in g++ 3.4.1!

      2004-08-06 <rjpeters>

        submitted this to gcc's bugzilla database along with a reduced
        testcase... the bug is trackable here:

                 http://gcc.gnu.org/bugzilla/show_bug.cgi?id=16905
   */
  for(int i = 0; i < cascadeChunk; i++)
  {
    cascadeImage[i].resize(setImageSizeX,setImageSizeY);
  }
#endif
}

/*
   OUTER LOOP/GROUP SUPRESSION COMPUTATION
   here we execute outer portions of the 6 layer loop in CINNIC
   for psuedo-convolution
   We also compute the group supression based upon change in groups
   overall total activity change (derivative)
*/
//#################################################################
void contourRun::calcGroups(std::vector< Image<float> > &imageMap,
                                 ContourNeuronCreate<float> &N, int iter)
{
  float hold;
  SMI[iter].resize(setImageSizeX,setImageSizeY,true);
  cascadeMap[iter].resize(setImageSizeX,setImageSizeY,true);
  GroupMap[iter].resize(setImageSizeX,setImageSizeY,true);
  for(int thing = 0; thing < Groups; thing++)
  {
    GroupHold[thing] = 0;
  }

  //First check potentials for each column at this iteration,
  //reset if nessessary
  for(int i = 0; i < setImageSizeX; i++)
  {
    for(int j = 0; j < setImageSizeY; j++)
    {
      for(int a = 0; a < AnglesUsed; a++)
      {
        //Add up all charges in this column
        //if charge is negative then make zero
        if(NeuronMatrix[iter][a][i][j].getCharge() < 0)
        {
          NeuronMatrix[iter][a][i][j].ResetCharge();
        }
        // add this iteration plus all others combined here
        // i.e. here lies the integrator for each pixel!
        hold = SM.getVal(i,j)+
          NeuronMatrix[iter][a][i][j].getCharge();
          SM.setVal(i,j,hold);

      }

      // leak this neuron
      hold = SM.getVal(i,j) - leak;
      if(hold < 0){hold = 0;}
      SM.setVal(i,j,hold);
      // Compute sigmoid of column this iteration
      hold = preSigmoid(SM.getVal(i,j),upperLimit);
      //hold = sigmoid2(upperLimit,SM.getVal(i,j));
      // set output (for movie) to a value from 0 to 256
      GroupMap[iter].setVal(i,j,hold);
      // set value into this iteration normalized
      if((i > 0) && (j > 0))
      {
        SMI[iter].setVal((i-1),(j-1),((hold/upperLimit)*255));
      }
      // calculate group DELTA if group adaptation selected
      if((iter > 0) && (adaptType == 1))
      {
        int ghold = (int)Group.getVal(i,j); //what group is this column in?
        int ihold = iter - 1; //what was the last iteration
        // find delta value for sig. on column, add to group
        GroupHold[ghold] += GroupMap[iter].getVal(i,j) -
          GroupMap[ihold].getVal(i,j);
      }
    }
  }

  // modify supression values using groups
  if(adaptType == 1)
  {
    for(int g = 0; g < Groups; g++)
    {
      // if values are too big then add suppression
      // this is done per pixel group
      //LINFO("Group %d DELTA %f \tSUPRESSION %f",g,GroupHold[g],GroupMod[g]);
      if(GroupHold[g] > GroupTop)
      {
        GroupMod[g] += supressionAdd*(GroupHold[g]-GroupTop);
        GroupMod2[g] = 1/GroupMod[g];
        //LINFO("NEW Group Supression %d:%f",g,GroupMod[g]);
      }
      // if values are too small then remove supression
      if(GroupHold[g] < GroupBottom)
      {
        GroupMod[g] -= supressionSub;
      }
    }
  }
}

//#################################################################
void contourRun::runImageSigmoid(std::vector< Image<float> > &imageMap,
                                 ContourNeuronCreate<float> &N, int iter)
{
  calcGroups(imageMap,N,iter);
  iterateConvolve(iter,imageMap,N);
}


//#################################################################
void contourRun::iterateConvolve(int iter,std::vector< Image<float> > &imageMap,
                                 ContourNeuronCreate<float> &N,
                                 const int node)
{
  //RUN hypercolumn get charges for i put charges for i+1
  for(int a = 0; a < AnglesUsed; a++) //this neuron's angle
  {
    for(int b = 0; b < AnglesUsed; b++) //other neuron's angle
    {
      convolveSimple(iter,imageMap,N,a,b);
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
void contourRun::convolveSimple(int iter,std::vector< Image<float> > &imageMap,
                                ContourNeuronCreate<float> &N,
                                const int a, const int b, const int node)
{
  for(int i = 0; i < setImageSizeX; i++) //This Neuron's position X
  {
    for(int j = 0; j < setImageSizeY; j++) //This Neuron's position Y
    {
      if(imageMap[a].getVal(i,j) > 0.001F) //optimization
      {

        //MUST KILL OTHER NEURONS!!!! - fast placicity
        //sigmoid this at some point ?
        float mod;
        mod = NeuronMatrix[iter][a][i][j].getCharge()*fastPlast;
        if(mod < 1){mod = 1;}
        if(mod > 5){mod = 5;}
        int thisGroup = (int)Group.getVal(i,j);
        float crap = imageMap[a].getVal(i,j)*
          (passThroughGain/((GroupMod[thisGroup]*5)-4));
        NeuronMatrix[(iter+1)][a][i][j].
          ChargeSimple(crap);
        for(int k = 0; k <= XSize; k++) //Other Neuron's position X
        {
          //Insp2X = (InspX-i) + (int)XCenter;
          //Current position plus its field - center
          InspX = i + (k-(int)XCenter);
          if(InspX >= 0)
          {
            if(InspX < setImageSizeX)
            {
              for(int l = 0; l <= YSize; l++) //Other Neuron's position Y
              {
                InspY = j - (l-(int)YCenter);
                //stay inside of image
                if(InspY >= 0)
                {
                  if(InspY < setImageSizeY)
                  {
                    //check that this element != 0
                    if(N.FourDNeuralMap[a][b][k][l].zero)
                    {
                      //LINFO("%d,%d",InspX,InspY);
                      //optimization
                      if(imageMap[b].getVal(InspX,InspY) > orThresh )
                      {
                        //Insp2Y = (InspY-j) + (int)YCenter; // <-- FIX?
                        float hold;
                        //this is done this way to optimize the code
                        //for multiple
                        //iteration, by storing some of the first iterations
                        //results in an array
                        if(iter == 0)
                        {
                          //## it is important to note that ImageOpt
                          //= Image*energy, thus it's an optimization
                          //if this is a negative number ,
                          //then supression may be modified

                          //FORMULA HERE
                          //orient filered image1 * orient filtered image2
                          //* their excitation

                          storeVal[iterCounter] = ((imageOpt[a].getVal(i,j) *
                             imageMap[b].getVal(InspX,InspY)) *
                             N.FourDNeuralMap[a][b][k][l].angABD);
                        }
                        if(N.FourDNeuralMap[a][b][k][l].angABD < 0)
                        {

                          hold =
                            GroupMod[thisGroup] * mod * storeVal[iterCounter];
                        }
                        else
                        {
                          hold = mod * storeVal[iterCounter];
                        }
                        //set energy in sal map
                        NeuronMatrix[(iter+1)][b][InspX][InspY].
                          ChargeSimple(hold);
                        iterCounter++;
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
void contourRun::contourRunMain(std::vector< Image<float> > &imageMap,
                                ContourNeuronCreate<float> &N
                                ,readConfig &config, Image<float> &group,
                                int groups,int iter,float groupTop)
{

  if(iter == 0)
  {
    GroupTop = groupTop;
    setConfig(config);
    setImageSize(imageMap[1].getWidth(),imageMap[1].getHeight());
    resetMatrix();
    preImage(imageMap,N);
    Group = group;
    // groups can be set to either 1 or 0 depending on whether or not you
    // want supression to happen only when a group reaches threshold
    // excitation or not.
    GroupMod.resize(groups,initialGroupVal);
    GroupMod2.resize(groups,initialGroupVal); //?
    GroupHold.resize(groups,0.0F);
    Groups = groups;
  }
  iterCounter = 0;
  runImageSigmoid(imageMap,N,iter);

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
