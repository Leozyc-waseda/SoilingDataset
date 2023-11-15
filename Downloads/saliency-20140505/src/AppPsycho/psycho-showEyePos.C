/*!@file AppPsycho/psycho-showEyePos.C Filter and Show the eye positions*/

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: $
// $Id: $


#include "BayesFilters/UKF.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Util/MathFunctions.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Psycho/EyeSFile.H"

//Create a tracker to track a particle with a position and velocity
class EyeFilter : public UKF
{
  public:
    EyeFilter() : 
      UKF(6, 2) //Initialize with pos,vel,acc as state and pos as observation
    {

      //Initial state
      itsState.setVal(0,0, 640/2);
      itsState.setVal(0,1, 480/2);
      itsState.setVal(0,2, 0);
      itsState.setVal(0,3, 0);
      itsState.setVal(0,4, 0);
      itsState.setVal(0,5, 0);

      //Initial covariance 
      itsSigma.setVal(0,0,100.0*100.0); //X Position
      itsSigma.setVal(1,1,100.0*100.0); //Y Position
      itsSigma.setVal(2,2,0.01*0.01); //X Velocity
      itsSigma.setVal(3,3,0.01*0.01); //Y Velocity
      itsSigma.setVal(4,4,0.01*0.01); //X Acc
      itsSigma.setVal(5,5,0.01*0.01); //Y Acc

      double posVar=4;
      double velVar=0.1;
      double accVar=0.1;
      //Initial noise matrix
      itsR.setVal(0,0,posVar); //X Pos
      itsR.setVal(1,1,posVar); //Y Pos
      itsR.setVal(2,2,velVar); //X Vel
      itsR.setVal(3,3,velVar); //Y Vel
      itsR.setVal(4,4,accVar); //X Acc
      itsR.setVal(5,5,accVar); //Y Acc
    }

   ~EyeFilter() {}; 

   Image<double> getNextState(const Image<double>& X, int k)
    {
      double posX = X.getVal(k,0);
      double posY = X.getVal(k,1);
      double velX = X.getVal(k,2);
      double velY = X.getVal(k,3);
      double accX = X.getVal(k,4);
      double accY = X.getVal(k,5);

      Image<double> Xnew(1,itsNumStates, ZEROS);
      Xnew[0] = posX + velX;
      Xnew[1] = posY + velY;
      Xnew[2] = velX + accX;
      Xnew[3] = velY + accY;
      Xnew[4] = accX;
      Xnew[5] = accY;

      return Xnew;
    }

   Image<double> getObservation(const Image<double>& X, int k)
    {
      double posX = X.getVal(k,0);
      double posY = X.getVal(k,1);


      Image<double> zNew(1,itsNumObservations, ZEROS);
      zNew[0] = posX;
      zNew[1] = posY;
      return zNew;
    }

   Point2D<double> getPos() { return Point2D<double>(itsState[0], itsState[1]); }
   Point2D<double> getVel() { return Point2D<double>(itsState[2], itsState[3]); }
   Point2D<double> getAcc() { return Point2D<double>(itsState[4], itsState[5]); }

   void getPosEllipse(Point2D<float>& mu, Point2D<float>& sigma)
    {
      //Set the Mean
      mu.i = itsState[0];
      mu.j = itsState[1];

      //Get the 95% uncertainty region
      //Set the major/minor axis
      sigma = Point2D<float>(2*sqrt(itsSigma.getVal(0,0)),
                             2*sqrt(itsSigma.getVal(1,1)));
    }


};


int main(int argc, char *argv[]){

  ModelManager manager("ShowEyePos");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<EyeSFile> eyeS(new EyeSFile(manager));
  manager.addSubComponent(eyeS);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<velocity threshold> <delay>", 2, 2) == false) return(1);
  // let's get all our ModelComponent instances started:
  manager.start();

  double velThresh = atof(manager.getExtraArg(0).c_str());
  int delay = atoi(manager.getExtraArg(1).c_str());
  


  EyeFilter eyeFilter; 


  double pVar=3.0; 
  Image<double> zNoise(2,2,ZEROS);
  zNoise.setVal(0,0,pVar*pVar);
  zNoise.setVal(1,1,pVar*pVar);
    

  Point2D<float> lastSaccLoc;

  Point2D<float> eyeLoc = eyeS->getPos();
  while(eyeLoc.isValid())
  {
    ////Predict the eye position
    eyeFilter.predictState();
    eyeFilter.predictObservation(zNoise);


    ////Update
    Image<double> z(1,2,ZEROS);
    z[0] = eyeLoc.i; z[1] = eyeLoc.j;
    eyeFilter.update(z, zNoise);
    

    //Get the predicted pos,vel and acc
    Point2D<double> pos = eyeFilter.getPos();
    Point2D<double> vel = eyeFilter.getVel();
    Point2D<double> acc = eyeFilter.getAcc();
    

    //Display the eye movement
    Image<PixRGB<byte> > display(eyeS->getRawInputDims(),ZEROS);

    //Draw the Real eye movement
    drawCircle(display, (Point2D<int>)eyeLoc, 2, PixRGB<byte>(0,255,0), 2);

    //Draw Our Predicted mean and covariance
    Point2D<float> mu, sigma;
    eyeFilter.getPosEllipse(mu,sigma);

    if (vel.magnitude() > velThresh)
    {
      drawCircle(display, (Point2D<int>)mu, 3, PixRGB<byte>(0,0,255), 3); 
      drawLine(display, (Point2D<int>)lastSaccLoc, (Point2D<int>)mu, PixRGB<byte>(255,255,255), 3);
    } else {
      drawCircle(display, (Point2D<int>)mu, 1, PixRGB<byte>(255,0,0), 1); 
      lastSaccLoc = mu;
    }


    if (sigma.i < 500 && sigma.j < 500)
      drawEllipse(display, (Point2D<int>)mu, sigma.i,sigma.j, PixRGB<byte>(255,0,0));
 
    //Show the Predicted Pos Vel and Acc

    char msg[255];
    sprintf(msg, "Pos: %0.1fx%0.1f", pos.i, pos.j);
    writeText(display, Point2D<int>(0,0), msg,
        PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));
    sprintf(msg, "Vel: %0.2f", vel.magnitude());
    writeText(display, Point2D<int>(0,20), msg,
        PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));
    sprintf(msg, "Acc: %0.2f", acc.magnitude());
    writeText(display, Point2D<int>(0,40), msg,
        PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));


    ofs->writeRGB(display, "EyeMovement", FrameInfo("EyeMovement", SRC_POS));
    
    eyeLoc = eyeS->getPos(); //Get the next position

    usleep(delay);
  }



  manager.stop();
  exit(0);

}


