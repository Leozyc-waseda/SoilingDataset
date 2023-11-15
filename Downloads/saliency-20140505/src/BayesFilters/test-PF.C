/*!@file BayesFilters/test-PF.C test the filter*/

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


#include "BayesFilters/ParticleFilter.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Util/MathFunctions.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Image/Point3D.H"




//Create a tracker to track a particle with a position and velocity
class ParticleTracker : public ParticleFilter
{
  public:
    ParticleTracker(int numStates, int numObservations, int numParticles) : 
      ParticleFilter(numStates, numObservations, numParticles)
    {

    }

   ~ParticleTracker() {}; 

   Image<double> getNextState(const Image<double>& X)
    {

      double processMean = -0.1;
      double processSigma = 0.075;
      double processScale = 0.4;

      double posX = X.getVal(0,0);

      processMean = processMean + ((posX - processMean)*processScale) +
        processSigma * gaussianRand();

      Image<double> Xnew(1,itsNumStates, ZEROS);
      Xnew[0] = processMean;

      return Xnew;
    }

   Image<double> getObservation(const Image<double>& X)
    {
      double posX = X.getVal(0,0);

      Image<double> zNew(1,itsNumObservations, ZEROS);
      zNew[0] = posX;
      return zNew;
    }

};

std::vector<int> getIntervals(int n, int polyDegree)
{
  std::vector<int> intervals(n+polyDegree+1);

  for (int j=0; j<=n+polyDegree; j++)
  {
    if (j<polyDegree)
      intervals[j]=0;
    else
      if ((polyDegree<=j) && (j<=n))
        intervals[j]=j-polyDegree+1;
      else
        if (j>n)
          intervals[j]=n-polyDegree+2;  // if n-t=-2 then we're screwed, everything goes to 0
  }

  return intervals;

}


double blend(int k, int t, std::vector<int>& intervals, double v)
{
  double value;

  if (t==1)            // base case for the recursion
  {
    if ((intervals[k]<=v) && (v<intervals[k+1]))
      value=1;
    else
      value=0;
  } else {
    if ((intervals[k+t-1]==intervals[k]) && (intervals[k+t]==intervals[k+1]))  // check for divide by zero
      value = 0;
    else
      if (intervals[k+t-1]==intervals[k]) // if a term's denominator is zero,use just the other
        value = (intervals[k+t] - v) / (intervals[k+t] - intervals[k+1]) * blend(k+1, t-1, intervals, v);
      else
        if (intervals[k+t]==intervals[k+1])
          value = (v - intervals[k]) / (intervals[k+t-1] - intervals[k]) * blend(k, t-1, intervals, v);
        else
          value = (v - intervals[k]) / (intervals[k+t-1] - intervals[k]) * blend(k, t-1, intervals, v) +
            (intervals[k+t] - v) / (intervals[k+t] - intervals[k+1]) * blend(k+1, t-1, intervals, v);
  }

  return value;



}


double getWeight(int k, double s)
{

  double weight = 0;

  if (0 <= s && s < 1)
    weight = (s*s)/2;
  else if (1 <= s && s < 2)
    weight = (3/4)-((s-(3/2))*(s-(3/2)));
  else if (2 <= s && s < 3)
    weight = ((s-3)*(s-3))/2;

  return weight;
}

void drawBSpline(std::vector<Point3D<float> >& controlPoints)
{

  int numOutput = 100; //The number of points to output
  int polyDegree = 3;

  std::vector<int> intervals = getIntervals(controlPoints.size()-1, polyDegree);
  double incr = 1.0/double(controlPoints.size()-1); //(double) (controlPoints.size()-1-polyDegree+2)/(numOutput-1);
  LINFO("Inc %f", incr);

  std::vector<Point3D<float> > points;
  double interval = 0;
  for(int outputIndex=0; outputIndex<numOutput; outputIndex++)
  {
    //Compute point
    Point3D<float> point(0,0,0);
    for(uint k=0; k<controlPoints.size(); k++)
    {
      double blendVal = blend(k,polyDegree, intervals, interval);
      //double blendVal = getWeight(k, interval-k);
      point.x += blendVal*controlPoints[k].x;
      point.y += blendVal*controlPoints[k].y;
      point.z += blendVal*controlPoints[k].z;
    }

    LINFO("Point %f %f", point.x, point.y);
    points.push_back(point);
    interval += incr;
  }
  //the last point is the control
  //points.push_back(controlPoints[0]);

  LINFO("Draw");
  Image<PixRGB<byte> > tmp(640,480,ZEROS);
  for(uint i=0; i<controlPoints.size(); i++)
  {
    Point2D<int> cLoc = Point2D<int>(controlPoints[i].x,controlPoints[i].y);
    drawCircle(tmp, cLoc, 3, PixRGB<byte>(0,255,0));
  }

  for(uint i=0; i<points.size()-10; i++)
  {
    Point2D<int> loc = Point2D<int>(points[i].x,points[i].y);
    Point2D<int> loc2 = Point2D<int>(points[(i+1)%points.size()].x,points[(i+1)%points.size()].y);
    drawLine(tmp, loc, loc2, PixRGB<byte>(255,0,0));
    //if (tmp.coordsOk(loc))
    //  tmp.setVal(loc, PixRGB<byte>(255,0,0));
  }
  SHOWIMG(tmp);


}


int main(int argc, char *argv[]){

  ModelManager manager("Test UKF");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  // let's get all our ModelComponent instances started:
  manager.start();
  ///B Spline
  std::vector<Point3D<float> > controlPoints;
  //controlPoints.push_back(Point3D<float>(10 , 100,0));
  //controlPoints.push_back(Point3D<float>(200, 100,0));
  //controlPoints.push_back(Point3D<float>(345, 300,0));
  //controlPoints.push_back(Point3D<float>(400, 250,0));
  //controlPoints.push_back(Point3D<float>(500, 550,0));
  //controlPoints.push_back(Point3D<float>(550, 150,0));
  //controlPoints.push_back(Point3D<float>(570, 50,0));
  //controlPoints.push_back(Point3D<float>(600, 100,0));

  controlPoints.push_back(Point3D<float>(50*2.75,50*5.25,0));
  controlPoints.push_back(Point3D<float>(50*3.28,50*7.07,0));
  controlPoints.push_back(Point3D<float>(50*6.61,50*7.08,0));
  controlPoints.push_back(Point3D<float>(50*6.85,50*5.19,0));
  controlPoints.push_back(Point3D<float>(50*5.29,50*4.00,0));
  controlPoints.push_back(Point3D<float>(50*5.67,50*1.28,0));
  controlPoints.push_back(Point3D<float>(50*4.12,50*1.28,0));
  controlPoints.push_back(Point3D<float>(50*4.33,50*4.04,0));


  drawBSpline(controlPoints);

  //Initialize tracker with 4 states and 2 observations and 1000 particles
  ParticleTracker pt(4,2, 1000);

  //Simulate a moving particle with a particuler velocity
  Image<PixRGB<byte> > worldImg(640,480,ZEROS);

  double processMean = -0.1;
  double processSigma =  0.4;
  double scaling =  0.075;



  for(uint t=0; t<100; t++)
  {
    double truePosition = processMean;
    //Move the particle
    processMean = processMean + ((truePosition - processMean)*scaling) +
      processSigma * pt.gaussianRand();

    pt.predictState();

    //Observe the state and update
    Image<double> z(1,1,ZEROS);
    z[0] = processMean + processSigma*pt.gaussianRand();
    pt.update(z);

    //Show the results
    //worldImg.setVal(Point2D<int>(particlePosition), PixRGB<byte>(0,255,0));
    //if (worldImg.coordsOk(Point2D<int>(pt.getPosition())))
    //  worldImg.setVal(Point2D<int>(pt.getPosition()), PixRGB<byte>(255,0,0));
    //Image<PixRGB<byte> > tmp = worldImg;
    //drawCircle(tmp, Point2D<int>(particlePosition), 3, PixRGB<byte>(0,255,0),3);
    //drawCircle(tmp, Point2D<int>(pt.getPosition()), 3, PixRGB<byte>(255,0,0), 3); 
    //ofs->writeRGB(tmp, "Particle Tracker", FrameInfo("Particle Tracker", SRC_POS));
    //usleep(100000);
    //SHOWIMG(tmp);
    getchar();
  }

  exit(0);

}


