/*!@file CINNIC/contourNeuron.C CINNIC classes */

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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourNeuron.C $
// $Id: contourNeuron.C 12074 2009-11-24 07:51:51Z itti $
//

#include "CINNIC/contourNeuron.H"
#include "Util/log.H"

#include <cmath>
#include <cstdlib>
#include <cstdio>

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################


float AngleDropOff,AngleSupress,SupMult,OrthA,OrthB,OrthMult,valueCutOff;
float coLinearCutOff;
float NeuronExSize,NeuronSupSize,NeuronSupStart,NeuronOrthSize;
float CoLinearDiff;
//! How many angles are detected via gabor filter
//float NeuralAngles[AnglesUsed]={0,45,90,135};
float NeuralAngles[AnglesUsed]={CINNIC_ANGLES};
float XCenter, YCenter, ExReach, SupReach,OrthReach;
float linearFactor2e,linearFactor3e,linearFactor2s,linearFactor3s;
float distanceSupFactor2,distanceSupFactor3,distanceExFactor2,distanceExFactor3;

//This method will return the ratio of XMax to X constrained from 0 to 1

template <class TC2>
float ContourNeuronCreate<TC2>::FindRatio(TC2& XMax, TC2& X)
{
  return ((XMax-X)/XMax);
}

//General Pythagorian for finding the length of a line

template <class TC2>
float ContourNeuronCreate<TC2>::FindDistance(TC2& PiX, TC2& PiY)
{
  return (sqrt((PiX*PiX)+(PiY*PiY)));
}

//This method when called will find the center pixel in the neuron, i.e.
// the neuron pixel itself

template <class TC2>
void ContourNeuronCreate<TC2>::FindCenter()
{
  XCenter = XSize/2;
  YCenter = YSize/2;
}

template <class TC2>
void ContourNeuronCreate<TC2>::FindReach()
{
  ExReach = NeuronExSize/2;
  SupReach = NeuronSupSize/2;
  OrthReach = NeuronOrthSize/2;
}

//This method will find the angle to a pixel from the center neuron

template <class TC2>
float ContourNeuronCreate<TC2>::FindAngleToRad(TC2& X,TC2& Y)
{

  //Formula is the inverse Tangent of X/Y for X and Y represent the distance from
  //The center. Angles are added on depending on the cartesian quadrant the
  //line is in

  hold = 0;
  if(Y > YCenter) //Quad I or IV
  {
    if(X > XCenter) //Quad I
    {
      hold = (atan(fabs(XCenter-X)/fabs(YCenter-Y)));
    }
    if(X < XCenter) //Quad IV
    {
      hold = (2*pi)-(atan(fabs(XCenter-X)/fabs(YCenter-Y)));
    }
    if(X == XCenter) //Vertical vector at 0 degrees
    {
      hold = 0;
    }
  }

  if(Y < YCenter) //Quad II or III
  {
    if(X > XCenter) //Quad II
    {
      hold = pi-(atan(fabs(XCenter-X)/fabs(YCenter-Y)));
    }
    if(X < XCenter) //Quad III
    {
      hold = pi+(atan(fabs(XCenter-X)/fabs(YCenter-Y)));
    }
    if(X == XCenter) //Vertical vector at 180 degrees
    {
      hold = pi;
    }
  }

  if(Y == YCenter) // Horozontal Vector
  {
    if(X > XCenter)
    {
      hold = pi/2;
    }
    if(X < XCenter)
    {
      hold = pi+(pi/2);
    }
  }
  return hold;
}

//Convert radians to degrees

template <class TC2>
float ContourNeuronCreate<TC2>::FindRadToDeg(TC2& r)
{
  return ((180/pi) * r);
}

//find a linear function

template <class TC2>
float ContourNeuronCreate<TC2>::FindLinear(TC2& m, TC2& n,TC2& o,TC2& x, TC2& b)
{
  return ((m*x)+(n*(pow(x,2)))+(o*(pow(x,3)))+b);
}

template <class TC2>
float ContourNeuronCreate<TC2>::sigmoid(TC2& beta, TC2& v)
{
  return (1/(1+pow(2.71828,(-2.0*(beta * v)))));
}
//This method will return the generic neuron once completed

#if 0
template <class TC2>
PropHold ContourNeuronCreate<TC2>::GetCompleteNeuron()
{
  /* NOTE: g++4.4 warns
     src/CINNIC/contourNeuron.C: In member function ‘PropHold ContourNeuronCreate<TC2>::GetCompleteNeuron() [with TC2 = float]’:
     src/CINNIC/contourNeuron.C:187: warning: array subscript is above array bounds
  */

  return FourDNeuralMap[AnglesUsed][AnglesUsed][XSize][YSize];
}
#endif

//This method will create the contour neuron

template <class TC2>
void ContourNeuronCreate<TC2>::CreateNeuron(readConfig &config)
{
  one = 1;
  AngleDropOff = config.getItemValueF("AngleDropOff");
  AngleSupress = config.getItemValueF("AngleSupress");
  SupMult = config.getItemValueF("SupMult");
  OrthMult = config.getItemValueF("OrthMult");
  OrthA = config.getItemValueF("AngleOrthRangeA");
  OrthB = config.getItemValueF("AngleOrthRangeB");
  NeuronExSize = config.getItemValueF("NeuronExSize");
  NeuronSupSize = config.getItemValueF("NeuronSupSize");
  NeuronSupStart = config.getItemValueF("NeuronSupStart");
  NeuronOrthSize = config.getItemValueF("NeuronOrthSize");
  valueCutOff = config.getItemValueF("valueCutOff");
  CoLinearDiff = config.getItemValueF("CoLinearDiff");
  coLinearCutOff =  config.getItemValueF("coLinearCutOff");
  linearFactor2e = config.getItemValueF("linearFactor2e");
  linearFactor3e =  config.getItemValueF("linearFactor3e");
  linearFactor2s = config.getItemValueF("linearFactor2s");
  linearFactor3s =  config.getItemValueF("linearFactor3s");
  distanceSupFactor2 = config.getItemValueF("distanceSupFactor2");
  distanceSupFactor3 = config.getItemValueF("distanceSupFactor3");
  distanceExFactor2 = config.getItemValueF("distanceExFactor2");
  distanceExFactor3 = config.getItemValueF("distanceExFactor3");

  factor = -1/(1-FindRatio(foo = AngleCenter,bar = AngleDropOff));
  factor2 = -1/(1-FindRatio(foo = AngleCenter,bar = AngleSupress));
  OFactorA = -1/(1-FindRatio(foo = AngleCenter, bar = OrthA));
  OFactorB = -1/(1-FindRatio(foo = AngleCenter, bar = OrthB));
  FindCenter();
  FloatAngleCenter = AngleCenter;
  FindReach();
  //LINFO("%f %d %d %f %f factor", factor,AngleCenter,AngleDropOff,foo,bar);
  //comes out to -1/.3 here
  //iterate through each element in the neuron matrix and set that elements value
  LINFO("Starting neuron generation");
  for (int i = 0; i < AnglesUsed; i++)
  {
    printf(".");
    for (int j = 0; j < AnglesUsed; j++)
    {
      for (int k = 0; k <= XSize; k++)
      {
        for (int l = 0; l <= YSize; l++)
        {
          if((k == (int)XCenter) && (l == (int)YCenter)) //center is always 0
          {
            FourDNeuralMap[i][j][k][l].zero = false;
            FourDNeuralMap[i][j][k][l].angABD = 0;
            FourDNeuralMap[i][j][k][l].angAB = 0;
            FourDNeuralMap[i][j][k][l].dis = 0;
          }
          else
          {
            ii = i;jj = j;kk = k;ll = l;
            //Find the distance to another neuron
            distanceX = fabs(XCenter-kk);
            distanceY = fabs(YCenter-ll);
            distance =  FindDistance(distanceX, distanceY); //input distance from
            //center of pixel

            // Find distance ratio from 1 to 0 to a pixel for
            // (a) Full template size
            // (b) Full excitation reach
            // (c) Full supression reach
            // take linear distance plus a 2nd and 3rd order term if needed
            float exreach =  FindRatio(ExReach, distance);
            float supreach = FindRatio(SupReach, distance);
            FourDNeuralMap[i][j][k][l].dis = FindRatio(XCenter, distance);
            FourDNeuralMap[i][j][k][l].Edis = exreach + (distanceExFactor2*pow(exreach,2))
              + (distanceExFactor3*pow(exreach,3));
            FourDNeuralMap[i][j][k][l].Sdis = supreach + (distanceSupFactor2*pow(supreach,2))
              + (distanceSupFactor3*pow(supreach,3));
            FourDNeuralMap[i][j][k][l].Odis = FindRatio(OrthReach, distance);
            if(FourDNeuralMap[i][j][k][l].dis < 0)
              {FourDNeuralMap[i][j][k][l].dis = 0;}
            if(FourDNeuralMap[i][j][k][l].Edis < 0)
              {FourDNeuralMap[i][j][k][l].Edis = 0;}
            if(FourDNeuralMap[i][j][k][l].Sdis < 0)
              {FourDNeuralMap[i][j][k][l].Sdis = 0;}
            if(FourDNeuralMap[i][j][k][l].Odis < 0)
              {FourDNeuralMap[i][j][k][l].Odis = 0;}

            //find to what degree another neuron "points" at this neuron
            Angle = FindRadToDeg(hold = FindAngleToRad(kk,ll));

            //find the angle to the
            //neuron in degrees

            //Find the angles complement i.e. find its alignement
            //with an angle < 180
            if(Angle >= 180)
            {
              AngleA = Angle - 180;
            }
            else
            {
              AngleA = Angle;
            }

            //Find the difference
            //between the angle pointing the the other
            //neuron and the alignment of this neuron
            AngleAlpha = fabs(AngleA - NeuralAngles[i]);

            //Find the difference
            //between the angle pointing the the other
            //neuron and the alignment of that neuron
            //This aligns the polarity of the neural map
            //This neuron will only pass energy onto another neuron of the same
            //polarity. Neurons on one side
            //of a specific polarity can only send energy while the other side
            //can only recieve
            AngleBeta = fabs(AngleA - NeuralAngles[j]);

            //These numbers are shifted by 1 to catch floating point precision
            //errors
            FourDNeuralMap[i][j][k][l].coLinear = false;
            if((Angle <= (NeuralAngles[i]+90+coLinearCutOff))
               && (Angle >= (NeuralAngles[i]+90-coLinearCutOff)))
            {
               FourDNeuralMap[i][j][k][l].coLinear = true;
            }
            if((Angle <= (NeuralAngles[i]-90.0F+coLinearCutOff))
               && (Angle >= (NeuralAngles[i]-90.0F-coLinearCutOff)))
            {
               FourDNeuralMap[i][j][k][l].coLinear = true;
            }
            if((Angle <= (NeuralAngles[i]+270.0F+coLinearCutOff))
               && (Angle >= (NeuralAngles[i]+270.0F-coLinearCutOff)))
            {
               FourDNeuralMap[i][j][k][l].coLinear = true;
            }


            //---polarization effect---
            //Forward polarity
            if((Angle <= (NeuralAngles[i]+1))
               || (Angle >= (NeuralAngles[i]+179)))
            {
              FourDNeuralMap[i][j][k][l].pol = true;

              //This neuron can send energy
              if(((NeuralAngles[i] <= 90) &&
                  ((Angle < NeuralAngles[i]+270)
                   && (Angle >= NeuralAngles[i]+180)))
                 || ((NeuralAngles[i] > 90) &&
                     ((Angle >= NeuralAngles[i]+180) ||
                      (Angle < NeuralAngles[i]-90))))
              {
                FourDNeuralMap[i][j][k][l].sender = true;
              }
              // This neuron can only recieve
              else
              {
                FourDNeuralMap[i][j][k][l].sender = false;
              }
            }
            //Reverse polarity
            else
            {
              FourDNeuralMap[i][j][k][l].pol = false;
              //This neuron can only recieve
              if((Angle > NeuralAngles[i]+90)
                 || (Angle >= (NeuralAngles[i]+270)))

              {
                FourDNeuralMap[i][j][k][l].sender = false;
              }
              // This neuron can send energy
              else
              {
                FourDNeuralMap[i][j][k][l].sender = true;
              }
            }

            //Find the semetric difference between angles AngleCenter is more then
            //likely 90 degrees
            AngleAlpha = AngleCenter - fabs(AngleAlpha - AngleCenter);
            AngleBeta = AngleCenter - fabs(AngleBeta - AngleCenter);

            phi = fabs(NeuralAngles[i]-NeuralAngles[j]);
            theta = fabs(90-AngleAlpha);
            OT = 90-theta;
            //find if orthogonal values need to be calculated and calculate their
            //value
            stop = false;
            FourDNeuralMap[i][j][k][l].orth = false;

            // find the roughly orthogonal oriented neurons ratios
            // this if statement is true for orthogonal pixels within
            // the set range
            if((OT < OrthA) && ((phi > (90-OrthB)) && (phi < (90+OrthB)))) // <<<BUG?
            {
              FourDNeuralMap[i][j][k][l].ang2 = FindRatio(OrthA,OT) * OrthMult;
              FourDNeuralMap[i][j][k][l].angABD =
                FourDNeuralMap[i][j][k][l].ang2 *
                FourDNeuralMap[i][j][k][l].Odis;
              stop = true;
              FourDNeuralMap[i][j][k][l].zero = true;
              FourDNeuralMap[i][j][k][l].orth = true;
            }
            else
            {
              FourDNeuralMap[i][j][k][l].ang2 =
                FindRatio(FloatAngleCenter,AngleAlpha);
              FourDNeuralMap[i][j][k][l].ang3 =
                1-FourDNeuralMap[i][j][k][l].ang2;
              // FIND LINEAR ANG2
              FourDNeuralMap[i][j][k][l].ang2 =
                FindLinear(factor,linearFactor2e,linearFactor3e,
                           FourDNeuralMap[i][j][k][l].ang2,one);
            }
            //find the angle beta that represents the difference
            // Between the angle to the other neuron and the alignment of that neuron
            FourDNeuralMap[i][j][k][l].ang =
              FindRatio(FloatAngleCenter,AngleBeta);
            // Find the difference between the idea angle which points
            // at the neuron and where this neuron actually points

            // determine this matrix for not angle greater then 30 deg.
            FourDNeuralMap[i][j][k][l].zero = false;

            // set 0 or supression angles (e.g. set the value for supression angles)
            if(FourDNeuralMap[i][j][k][l].ang2 < 0)
            {
              // FIND LINEAR ANG2
              FourDNeuralMap[i][j][k][l].ang2 =\
                -1*FindLinear(factor2,linearFactor2s,linearFactor3s,
                              FourDNeuralMap[i][j][k][l].ang3,one);
              if(FourDNeuralMap[i][j][k][l].ang2 > 0)
              {
                FourDNeuralMap[i][j][k][l].ang2 = 0;
              }
            }
            // FIND LINEAR ANG
            FourDNeuralMap[i][j][k][l].ang =\
            FindLinear(factor,linearFactor2e,linearFactor3e,
                       FourDNeuralMap[i][j][k][l].ang,one);

            //set 0 or FIND supression angles (e.g. set the value for supression angles)
            //else FIND excitation angles (Edis -> excitation, Sdis -> supression)
            if(FourDNeuralMap[i][j][k][l].ang2 < 0)
            {
              float diff;
              diff = fabs(NeuralAngles[i]-NeuralAngles[j]);
              if((diff < CoLinearDiff) || (diff > (180 - CoLinearDiff))) //<<fixed
              {
                //Set Alpha/Beta combo supression if no orthogonal exception?
                if(diff > (180 - CoLinearDiff)) // << This fixed the 0/165 bug
                {
                  diff = 180 - diff;
                }
                FourDNeuralMap[i][j][k][l].angAB =
                  FourDNeuralMap[i][j][k][l].ang2 *
                  FindRatio(CoLinearDiff,diff);

                // Find supression for angABD
                if((FourDNeuralMap[i][j][k][l].Sdis > 0)
                   && (distance >= NeuronSupStart))
                {
                  FourDNeuralMap[i][j][k][l].angABD = SupMult*
                    (((-1*FourDNeuralMap[i][j][k][l].Sdis)+
                      FourDNeuralMap[i][j][k][l].angAB)/2);
                FourDNeuralMap[i][j][k][l].zero = true;
                }
              }
            }
            //eliminate very small amplifications then derive angABD (OR set excitation angles here)
            else
            {
              if((FourDNeuralMap[i][j][k][l].ang > valueCutOff)
                 && (stop == false))
              {
                //Set alpha/beta combo for excitation
                FourDNeuralMap[i][j][k][l].angAB = \
                 (FourDNeuralMap[i][j][k][l].ang2*
                  FourDNeuralMap[i][j][k][l].ang);

                if((FourDNeuralMap[i][j][k][l].angAB > valueCutOff) && \
                   (FourDNeuralMap[i][j][k][l].Edis > valueCutOff))
                {
                  FourDNeuralMap[i][j][k][l].angABD = \
                   ((FourDNeuralMap[i][j][k][l].angAB
                     +FourDNeuralMap[i][j][k][l].Edis)/2);
                  if(FourDNeuralMap[i][j][k][l].angABD > valueCutOff)
                  {
                    //ignore small multipliers
                    FourDNeuralMap[i][j][k][l].zero = true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  printf("\n");
}

template class ContourNeuronCreate<float>;

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
