/*!@file SceneUnderstanding/LineGrouping.C  */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/LineGrouping.C $
// $Id: LineGrouping.C 14181 2010-10-28 22:46:20Z lior $
//

#ifndef LineGrouping_C_DEFINED
#define LineGrouping_C_DEFINED

#include "plugins/SceneUnderstanding/LineGrouping.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/Convolutions.H"
#include "Image/MatrixOps.H"
#include "Simulation/SimEventQueue.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <stack>

const ModelOptionCateg MOC_LineGrouping = {
  MOC_SORTPRI_3,   "LineGrouping-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_LineGroupingShowDebug =
  { MODOPT_ARG(bool), "LineGroupingShowDebug", &MOC_LineGrouping, OPTEXP_CORE,
    "Show debug img",
    "linegrouping-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(LineGrouping);

std::vector<Point2D<double> > getVel(const std::vector<Point2D<int> >& lines)
{
  std::vector<Point2D<double> > vel;

  for(uint i=0; i<lines.size()-1; i++)
  {
    Point2D<int> dPos = lines[i+1]-lines[i];
    double mag = sqrt((dPos.i*dPos.i) + (dPos.j*dPos.j))/4;
    for(int j=0; j<int(mag+0.5); j++)
      vel.push_back(Point2D<double>(dPos/mag));
  }

  return vel;

}

int quantize(float x, float y, float z)
{
    int val = 0;

    //Get the magnitude 
    double rho = sqrt(x*x+y*y+z*z);

    if (rho>3.0)
      val = 3<<3;
    else if (rho > 2.0)
      val = 2<<3;
    else if (rho>1.0)
      val = 1<<3;
    else
      val = 0;

    if (x>y) val |= 1<<2; 
    if (y>z) val |= 1<<1; 
    if (z>x) val |= 1; 

    return val;
}



// ######################################################################
LineGrouping::LineGrouping(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV2Output),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_LineGroupingShowDebug, this)
  
{
  //Add an hmm with 5 states and 32 possible observations
  std::vector<uint> states; //5 States
  for(uint i=0; i<5; i++) 
    states.push_back(i);

  std::vector<uint> posibleObservations; //32
  for(uint i=0; i<32; i++) 
    posibleObservations.push_back(i);

 itsHMM = HMM<uint>(states, posibleObservations, "Applelogo");

  std::vector<Point2D<int> > lines;
  lines.push_back(Point2D<int>(7, 71));
  lines.push_back(Point2D<int>(17, 88));
  lines.push_back(Point2D<int>(19, 90));
  lines.push_back(Point2D<int>(27, 95)); 
  lines.push_back(Point2D<int>(29, 95));
  lines.push_back(Point2D<int>(38, 91));
  lines.push_back(Point2D<int>(39, 91));
  lines.push_back(Point2D<int>(55, 95));
  lines.push_back(Point2D<int>(56, 95));
  lines.push_back(Point2D<int>(61, 93));
  lines.push_back(Point2D<int>(62, 93));
  lines.push_back(Point2D<int>(74, 73)); 
  lines.push_back(Point2D<int>(74, 72));
  lines.push_back(Point2D<int>(65, 62));
  lines.push_back(Point2D<int>(65, 61));
  lines.push_back(Point2D<int>(63, 57));
  lines.push_back(Point2D<int>(63, 56));
  lines.push_back(Point2D<int>(65, 42));
  lines.push_back(Point2D<int>(66, 42));
  lines.push_back(Point2D<int>(71, 36));
  lines.push_back(Point2D<int>(70, 35));
  lines.push_back(Point2D<int>(68, 33));
  lines.push_back(Point2D<int>(67, 32));
  lines.push_back(Point2D<int>(53, 29));
  lines.push_back(Point2D<int>(52, 30));
  lines.push_back(Point2D<int>(45, 32));
  lines.push_back(Point2D<int>(44, 27));
  lines.push_back(Point2D<int>(56, 15));
  lines.push_back(Point2D<int>(56, 14));
  lines.push_back(Point2D<int>(57, 7));
  lines.push_back(Point2D<int>(56, 6));
  lines.push_back(Point2D<int>(53, 7));
  lines.push_back(Point2D<int>(52, 7));
  lines.push_back(Point2D<int>(40, 19)); 
  lines.push_back(Point2D<int>(40, 20));
  lines.push_back(Point2D<int>(42, 26));
  lines.push_back(Point2D<int>(44, 33));
  lines.push_back(Point2D<int>(25, 29));
  lines.push_back(Point2D<int>(24, 29));
  lines.push_back(Point2D<int>(17, 31));
  lines.push_back(Point2D<int>(15, 32));
  lines.push_back(Point2D<int>(6, 43));
  lines.push_back(Point2D<int>(5, 45));
  lines.push_back(Point2D<int>(5, 64));
  lines.push_back(Point2D<int>(6, 65));
  lines.push_back(Point2D<int>(7, 70));

  //scale the lines
  for(uint i=0; i<lines.size(); i++)
    lines[i] *= 2;
 

  //Set the default transitions
  itsHMM.setStateTransition(0, 0, 0.5);
  itsHMM.setStateTransition(0, 1, 0.5);
  itsHMM.setStateTransition(1, 1, 0.5);
  itsHMM.setStateTransition(1, 2, 0.5);
  itsHMM.setStateTransition(2, 2, 0.5);
  itsHMM.setStateTransition(2, 3, 0.5);
  itsHMM.setStateTransition(3, 3, 0.5);
  itsHMM.setStateTransition(3, 4, 0.5);
  itsHMM.setStateTransition(4, 4, 1);

  //set the initial sstate
  itsHMM.setCurrentState(0, 1); //We start at the first state

  std::vector<Point2D<double> > vel = getVel(lines);

  std::vector< std::vector<uint> > observations;
  for(size_t j=0; j<vel.size(); j++)
  {
    std::vector<uint> observation;
    printf("InputValue ");
    for(size_t i=0; i<vel.size(); i++)
    {
      uint value = quantize(vel[(i+j)%vel.size()].i,
                          vel[(i+j)%vel.size()].j, 0);
      printf("%i ", value);
      observation.push_back(value);
    }
    printf("\n");
    observations.push_back(observation);
  }
  LINFO("Train");
  itsHMM.train(observations, 50);
  LINFO("Done");

}

// ######################################################################
LineGrouping::~LineGrouping()
{
}

// ######################################################################
void LineGrouping::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event --%s-- %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (strcmp(e->getWinName(), "LineGrouping"))
    return;

  switch(e->getKey())
  {
    case 111: //98: //111: //up
      break;
    case 116: //104: //116: //down
      break;
    case 113: //100: //113: //left
      break;
    case 114: //102: //114: //right
      break;
    case 21: //=
      break;
    case 20: //-
      break;
    case 38: //a
      break;
    case 52: //z
      break;
    case 39: //s
      break;
    case 53: //x
      break;
    case 40: //d
      break;
    case 54: //c
      break;
    case 10: //1
      break;
    case 24: //q
      break;
    case 11: //2
      break;
    case 25: //w
      break;
    case 12: //3
      break;
    case 26: //e
      break;
    case 13: //4
      break;
    case 27: //r
      break;
    case 14: //5
      break;
    case 28: //t
      break;
    case 15: //6
      break;
    case 29: //y
      break;
  }


  evolve(q);

}


// ######################################################################
void LineGrouping::onSimEventV2Output(SimEventQueue& q, rutz::shared_ptr<SimEventV2Output>& e)
{
  //Check if we have the smap
  //if (SeC<SimEventSMapOutput> smap = q.check<SimEventSMapOutput>(this))
  //  itsSMap = smap->getSMap();

  //Check if we have the corners
  itsLines = e->getLines();
  itsInputDims = e->getDims();

  evolve(q);

}

// ######################################################################
void LineGrouping::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage(q);
      if (disp.initialized())
        ofs->writeRgbLayout(disp, "LineGrouping", FrameInfo("LineGrouping", SRC_POS));
    }
}


// ######################################################################
void LineGrouping::evolve(SimEventQueue& q)
{

	Image<PixRGB<byte> > linesMask(itsInputDims, ZEROS);
  //Prepare an image with array of indexes to know where line are located quickly
  Image<std::vector<uint> > linesIndices(itsInputDims, NO_INIT);
  for(uint i=0; i<itsLines.size(); i++)
  {
    V2::LineSegment& ls = itsLines[i];

    Point2D<int> p1 = Point2D<int>(ls.p1);
    Point2D<int> p2 = Point2D<int>(ls.p2);
    linesIndices[p1].push_back(i);
    linesIndices[p2].push_back(i);
		drawLine(linesMask, p1, p2, PixRGB<byte>(50,50,50));
  }


  //Mark if the line has been used or not
	std::vector<uint> lineColour(itsLines.size(), 0);

  while(1)
  {
    //LinesGroup linesGroup;
    std::stack<LineInfo> linesStack;
		std::vector<LineInfo> linesGroup;
		std::vector<uint> linesGroupLevel;
		std::list<uint> linesGroupList;

    //Pick a line to group (sort by size)
		uint idx = 2;
		linesStack.push(LineInfo(idx));
		//uint stackLevel = 0;

    while(!linesStack.empty())
		{
      //Get the first line in the stack and pop it
			LineInfo lineInfo = linesStack.top();
      linesGroup.push_back(lineInfo);
			linesStack.pop();

      //Mark that line as used
			lineColour[lineInfo.idx] = 1;

			LINFO("Following line %d cost %f", lineInfo.idx, lineInfo.cost);
			Point2D<int> p1 = (Point2D<int>)itsLines[lineInfo.idx].p1;
			Point2D<int> p2 = (Point2D<int>)itsLines[lineInfo.idx].p2;
			drawLine(linesMask, p1, p2, PixRGB<byte>(0,255,0));

			int radius = 5;
      std::vector<LineInfo> lines;
			//First End Point
			//Find the nerest line endpoint to this line's p1 end point
      lines = getLocalLines(p1, radius, lineColour, linesIndices);
      std::vector<LineInfo> p2Lines = getLocalLines(p2, radius, lineColour, linesIndices);
      lines.insert(lines.end(), p2Lines.begin(), p2Lines.end());

      setTopDownCost(lines, linesGroup);




      //Sort by cost
      std::sort(lines.begin(), lines.end(), LineInfoCmp());


      //
      for(uint i=0; i<lines.size(); i++)
      {
        LineInfo line = lines[i];
        linesStack.push(line);

        //Show the line we are following
        Point2D<int> p1 = (Point2D<int>)itsLines[line.idx].p1;
        Point2D<int> p2 = (Point2D<int>)itsLines[line.idx].p2;

        LINFO("Found line %i at X %i y %i cost %f", line.idx, p1.i, p1.j, line.cost);
        drawLine(linesMask, p1, p2, PixRGB<byte>(255,0,0));
        SHOWIMG(linesMask);
      }

			LINFO("Lines added %zu", lines.size());
			if (lines.size() == 0)
			{
				//No more lines; dump the stack to the group
				SHOWIMG(linesMask);
				LINFO("Showing Stack");
				Image<PixRGB<byte> > tmp(itsInputDims, NO_INIT);
				for(uint i=0; i<itsLines.size(); i++)
				{
					V2::LineSegment& ls = itsLines[i];
					Point2D<int> p1 = Point2D<int>(ls.p1);
					Point2D<int> p2 = Point2D<int>(ls.p2);
					drawLine(tmp, p1, p2, PixRGB<byte>(50,50,50));
				}

				for(uint i=0; i<linesGroup.size(); i++)
				{
					LINFO("Stack %i %i %f", i, linesGroup[i].idx, linesGroup[i].cost);
					V2::LineSegment& ls = itsLines[linesGroup[i].idx];
					Point2D<int> p1 = Point2D<int>(ls.p1);
					Point2D<int> p2 = Point2D<int>(ls.p2);
					drawLine(tmp, p1, p2, PixRGB<byte>(0,255,0));
				}
				linesGroup.pop_back();
				
				SHOWIMG(tmp);
			} //else {
			//	stackLevel++;
			//	SHOWIMG(linesMask);
			//}
		}
		LINFO("Done");
    getchar();
  }
}


void LineGrouping::setTopDownCost(std::vector<LineInfo>& newLines,const std::vector<LineInfo>& contour)
{

  //Convert the current counter into velocities
  std::vector<Point2D<int> > locations;

  V2::LineSegment& ls = itsLines[contour[0].idx];
  Point2D<int> p1 = Point2D<int>(ls.p1);
  Point2D<int> p2 = Point2D<int>(ls.p2);
  locations.push_back(p1);
  locations.push_back(p2);

  for(uint i=1; i<contour.size(); i++)
  {
    V2::LineSegment& ls = itsLines[contour[i].idx];
    Point2D<int> p1 = Point2D<int>(ls.p1);
    Point2D<int> p2 = Point2D<int>(ls.p2);

    if (p2.distance(p1) > p2.distance(p2))
      locations.push_back(p1);
    else
      locations.push_back(p2);
  }


  Point2D<int> lastLoc = locations[locations.size()-2];
  //For each new line, check the probability that its in the model
  for(uint i=0; i<newLines.size(); i++)
  {
    //Add the new line
    std::vector<Point2D<int> > newLocations = locations;
    V2::LineSegment& ls = itsLines[newLines[i].idx];
    Point2D<int> p1 = Point2D<int>(ls.p1);
    Point2D<int> p2 = Point2D<int>(ls.p2);

    if (lastLoc.distance(p1) > lastLoc.distance(p2))
      newLocations.push_back(p1);
    else
      newLocations.push_back(p2);

    //Quantize the observations
    std::vector<Point2D<double> > vel = getVel(newLocations);
    std::vector<uint> observations; 
    for(size_t j=0; j<vel.size(); j++)
    {
      uint value = quantize(vel[j].i, vel[j].j, 0);
      observations.push_back(value);
    }
 
    //Set the prob from the top down model
    if (vel.size() > 10)
    {
      double prob = itsHMM.forward(observations);
      newLines[i].cost = prob;
      LINFO("Checking line %i idx %i cost %f", i, newLines[i].idx, prob);
    } else {
      LINFO("Follow bottom up line %i idx %i cost %f", i, newLines[i].idx, newLines[i].cost);
    }
    

    ////Show the Positions
    //Image<PixRGB<byte> > img(512,512, ZEROS);
    //Point2D<double> pos = (Point2D<double>)locations[0];
    //for(uint i=0; i<vel.size(); i++)
    //{
    //  if (img.coordsOk(Point2D<int>(pos)))
    //    img.setVal(Point2D<int>(pos), PixRGB<byte>(0,255,0));
    //  //LINFO("V: %f %f P: %f %f", vel[i].i, vel[i].j, pos.i, pos.j);
    //  pos += vel[i];
    //}
    //SHOWIMG(img);

  }

}




std::vector<LineGrouping::LineInfo>  LineGrouping::getLocalLines(const Point2D<int> loc,
    const int radius, 
    std::vector<uint>& lineColour,
    const Image<std::vector<uint> >& linesIndices)
{
  std::vector<LineInfo> lines;

  for(int x=loc.i-radius; x<=loc.i+radius; x++)
    for(int y=loc.j-radius; y<=loc.j+radius; y++)
    {
      Point2D<int> lineLoc =Point2D<int>(x,y); 

      if (linesIndices.coordsOk(lineLoc) &&
          linesIndices.getVal(lineLoc).size() > 0) //We have some lines, add them
      {
        for(uint j=0; j<linesIndices[lineLoc].size(); j++)
        {
          uint idx = linesIndices[lineLoc][j];
          if (!lineColour[idx])
          {
            lineColour[idx] = 1;
            double cost = loc.distance(lineLoc);
            lines.push_back(LineInfo(idx,cost));
          }
        }
      }
    }
  return lines;
}


std::vector<V2::LineSegment>  LineGrouping::getAndRemoveLinesNearLoc(Image<std::vector<uint> >& linesIndices,
    const Point2D<int> loc, const int radius)
{
  std::vector<V2::LineSegment> lines;

  for(int x=loc.i-radius; x<=loc.i+radius; x++)
    for(int y=loc.j-radius; y<=loc.j+radius; y++)
    {
      Point2D<int> lineLoc =Point2D<int>(x,y); 
      if (linesIndices.coordsOk(lineLoc) &&
          linesIndices.getVal(lineLoc).size() > 0) //We have some lines, add them
      {
        for(uint j=0; j<linesIndices[lineLoc].size(); j++)
        {
          uint idx = linesIndices[lineLoc][j];
          lines.push_back(itsLines[idx]);
        }
        linesIndices[lineLoc].clear(); //Remove that line from the list
      }
    }

  return lines;

}

Layout<PixRGB<byte> > LineGrouping::getDebugImage(SimEventQueue& q)
{
  Layout<PixRGB<byte> > outDisp;

  Image<PixRGB<byte> > inputImg(itsInputDims,ZEROS); 
  for(uint i=0; i<itsLines.size(); i++)
  {
    V2::LineSegment& ls = itsLines[i];
    drawLine(inputImg, Point2D<int>(ls.p1), Point2D<int>(ls.p2), PixRGB<byte>(255,0,0));
  }
  

  outDisp = inputImg; 

  return outDisp;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

