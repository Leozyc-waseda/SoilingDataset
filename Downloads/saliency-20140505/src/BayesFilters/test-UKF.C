/*!@file BayesFilters/test-UKF.C test the filter*/

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
#include "Image/Point3D.H"
#include "GUI/DebugWin.H"
#include "Util/MathFunctions.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"

//Create a tracker to track a particle with a position and velocity
class ParticleTracker : public UKF
{
  public:
    ParticleTracker() : 
      UKF(5, 2) //Position (x,y,theta) Vel and 2 observation
    {

      //Initial state
      itsState.setVal(0,0, 640/2);
      itsState.setVal(0,1, 480/2);
      itsState.setVal(0,2, 0);
      itsState.setVal(0,3, 0);
      itsState.setVal(0,4, 0);

      ////Initial covariance 
      itsSigma.setVal(0,0,150.0*150.0); //X Position
      itsSigma.setVal(1,1,150.0*150.0); //Y Position
      itsSigma.setVal(2,2,(1*M_PI/180)*(1*M_PI/180)); //Theta
      itsSigma.setVal(3,3,0.01*0.01); //Trans Velocity
      itsSigma.setVal(4,4,(1*M_PI/180)*(1*M_PI/180)); //Ang Velocity


      //Initial noise matrix
      double posVar=4.0;
      double angVar=1.0*M_PI/180;
      double tranVelVar=0.1;
      double angVelVar=0.1*M_PI/180;
      itsR.setVal(0,0,posVar*posVar);
      itsR.setVal(1,1,posVar*posVar);
      itsR.setVal(2,2,angVar*angVar);
      itsR.setVal(3,3,tranVelVar*tranVelVar);
      itsR.setVal(4,4,angVelVar*angVelVar);
    }

   ~ParticleTracker() {}; 

   Image<double> getNextState(const Image<double>& X, int k)
    {
      double posX = X.getVal(k,0);
      double posY = X.getVal(k,1);
      double ang =  X.getVal(k,2);
      double tranVel = X.getVal(k,3);
      double angVel = X.getVal(k,4);

      Image<double> Xnew(1,itsNumStates, ZEROS);
      double eps = 2.22044604925031e-16;

      double xc = posX - tranVel/(angVel+eps)*sin(ang);
      double yc = posY + tranVel/(angVel+eps)*cos(ang);

      posX = xc + tranVel/(angVel+eps)*sin(ang + angVel);
      posY = yc - tranVel/(angVel+eps)*cos(ang + angVel);
      ang += angVel;

      Xnew[0] = posX;
      Xnew[1] = posY;
      Xnew[2] = ang;
      Xnew[3] = tranVel;
      Xnew[4] = angVel;

      return Xnew;
    }

   Image<double> getObservation(const Image<double>& X, int k)
    {
      double posX = X.getVal(k,0);
      double posY = X.getVal(k,1);


      Image<double> zNew(1,itsNumObservations, ZEROS);
      zNew[0] = sqrt((posX*posX) + (posY*posY));
      zNew[1] = atan2(posY,posX);
      return zNew;
    }

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

  ModelManager manager("Test UKF");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  // let's get all our ModelComponent instances started:
  manager.start();

  ParticleTracker pt; //Initialize tracker with 4 states and 2 observations

  //Simulate a moving particle with a particular velocity
  Image<PixRGB<byte> > worldImg(640,480,ZEROS);

  Point3D<double> particlePosition(640/2,480/2-100,0); //Initial position
  
  Point3D<double> distractorPosition(640/2,480/2-100,0); //Initial position

  initRandomNumbersZero(); //For checking for correct values

  Point2D<int> lastParticlePosition;
  Point2D<int> lastSensorPosition;
  Point2D<int> lastPredictedPosition;
  for(uint t=0; t<360; t++)
  {
    double rangeNoise = 0; //(drand48()*8.0)-4.0;
    double angNoise = 0; //(drand48()*4.0)- 2.0;
    
    //Move the particle along a circular path

    //Center of circle
    double dR = 5+rangeNoise;
    double dA = (2 + angNoise)*M_PI/180;
    double eps = 2.22044604925031e-16;
    double distRatio = dR/(dA+eps);
    double xc = particlePosition.x - distRatio*sin(particlePosition.z);
    double yc = particlePosition.y + distRatio*cos(particlePosition.z);

    particlePosition.x = xc + distRatio*sin(particlePosition.z + dA);
    particlePosition.y = yc - distRatio*cos(particlePosition.z + dA);
    particlePosition.z += dA;

    //Distractor
    double dDR = -5-rangeNoise;
    double dDA = (-2 - angNoise)*M_PI/180;
    double distDRatio = dDR/(dDA+eps);
    double xdc = distractorPosition.x - distDRatio*sin(distractorPosition.z);
    double ydc = distractorPosition.y + distDRatio*cos(distractorPosition.z);

    distractorPosition.x = xdc + distDRatio*sin(distractorPosition.z + dDA);
    distractorPosition.y = ydc - distDRatio*cos(distractorPosition.z + dDA);
    distractorPosition.z += dDA;

    //Set the Sensor noise
    double zrn=10.0; 
    double zan=5*M_PI/180; //0.5 degrees uncertainty
    Image<double> zNoise(2,2,ZEROS);
    zNoise.setVal(0,0,zrn*zrn);
    zNoise.setVal(1,1,zan*zan);

    pt.predictState();
    pt.predictObservation(zNoise);


    //Compute the measurements
    Image<double> z1(1,2,ZEROS);
    z1[0] = sqrt(squareOf(particlePosition.x) + squareOf(particlePosition.y));
    z1[1] = atan2(particlePosition.y, particlePosition.x);

    z1[0] += 2; //(drand48()*16.0)- 8.0; 
    z1[1] += 2*M_PI/180; //((drand48()*5.0) - 2.5)*M_PI/180;

    Image<double> z2(1,2,ZEROS);
    z2[0] = sqrt(squareOf(distractorPosition.x) + squareOf(distractorPosition.y));
    z2[1] = atan2(distractorPosition.y, distractorPosition.x);

    z2[0] += 3; //(drand48()*10.0)- 5.0; 
    z2[1] += 3*M_PI/180; //((drand48()*3.0) - 1.5)*M_PI/180;

    double z1Prob = pt.getLikelihood(z1, Image<double>());
    double z2Prob = pt.getLikelihood(z2, Image<double>());

    Image<double> zUsed;
    if (z1Prob > 1.0e-5)
      zUsed=z1;
    if (z2Prob > 1.0e-5)
      zUsed=z1;
      

    //Store prediction for latter display
    Point2D<float> muP, sigmaP;
    pt.getPosEllipse(muP,sigmaP);

    pt.update(zUsed, zNoise);

    Point2D<float> mu, sigma;
    pt.getPosEllipse(mu,sigma);

    LINFO("True Pos: %f,%f predicted Pos: %f,%f Likelihood: z1 = %f z2 = %f",
        particlePosition.x, particlePosition.y,
        mu.i, mu.j,
        z1Prob, z2Prob);

    //Show the results
    Image<PixRGB<byte> > tmp = worldImg;

    //Draw the real particle position
    drawCircle(tmp, Point2D<int>(particlePosition.x, particlePosition.y), 2, PixRGB<byte>(0,255,0),2);
    drawCircle(tmp, Point2D<int>(distractorPosition.x, distractorPosition.y), 2, PixRGB<byte>(0,255,255),2);
    //Show the sensor
    Point2D<int> sensorPos(zUsed[0]*cos(zUsed[1]), zUsed[0]*sin(zUsed[1]));
    drawCircle(tmp, sensorPos, 2, PixRGB<byte>(0,0,255),2);

    //Show the predicted region
    drawCircle(tmp, (Point2D<int>)muP, 1, PixRGB<byte>(255,255,0), 1); 
    if (sigmaP.i < 500 && sigmaP.j < 500)
      drawEllipse(tmp, (Point2D<int>)muP, sigmaP.i,sigmaP.j, PixRGB<byte>(255,255,0));

    drawCircle(tmp, (Point2D<int>)mu, 1, PixRGB<byte>(255,0,0), 1); 
    if (sigma.i < 500 && sigma.j < 500)
      drawEllipse(tmp, (Point2D<int>)mu, sigma.i,sigma.j, PixRGB<byte>(255,0,0));


    //Draw history traces
    if (!lastParticlePosition.isValid())
      lastParticlePosition = Point2D<int>(particlePosition.x, particlePosition.y);
    drawLine(worldImg, lastParticlePosition,
        Point2D<int>(particlePosition.x, particlePosition.y),
        PixRGB<byte>(0,255,0));
    lastParticlePosition = Point2D<int>(particlePosition.x,
        particlePosition.y);

    if (!lastSensorPosition.isValid())
      lastSensorPosition = sensorPos;
    drawLine(worldImg, lastSensorPosition, sensorPos, PixRGB<byte>(0,0,255));
    lastSensorPosition = sensorPos;

    if (!lastPredictedPosition.isValid())
      lastPredictedPosition = (Point2D<int>)mu;
    drawLine(worldImg, lastPredictedPosition, (Point2D<int>)mu, PixRGB<byte>(255,0,0));
    lastPredictedPosition = (Point2D<int>)mu;

    ofs->writeRGB(tmp, "Particle Tracker", FrameInfo("Particle Tracker", SRC_POS));
    usleep(100000);
  }

  manager.stop();
  exit(0);

}


