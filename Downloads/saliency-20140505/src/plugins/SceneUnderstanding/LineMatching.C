/*!@file plugins/SceneUnderstanding/LineMatching.C  */

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
// $HeadURL: $
// $Id: $
//

#ifndef LineMatching_C_DEFINED
#define LineMatching_C_DEFINED

#include "Image/OpenCVUtil.H" //Needs to be first 

#include "plugins/SceneUnderstanding/LineMatching.H"

#include "Image/DrawOps.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueue.H"
#include "GUI/DebugWin.H"
#include "plugins/SceneUnderstanding/LMLineMatcher/LMLineMatcher.h"

#include "Media/MediaSimEvents.H"
//#include "Neuro/NeuroSimEvents.H"
#include "Transport/FrameInfo.H"
#include "Transport/FrameOstream.H"


const ModelOptionCateg MOC_LineMatching = {
  MOC_SORTPRI_3,   "LineMatching-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_LineMatchingShowDebug =
  { MODOPT_ARG(bool), "LineMatchingShowDebug", &MOC_LineMatching, OPTEXP_CORE,
    "Show debug img",
    "linematching-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(LineMatching);


// ######################################################################
LineMatching::LineMatching(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV2Output),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_LineMatchingShowDebug, this)
{
}

// ######################################################################
LineMatching::~LineMatching()
{
}


// ######################################################################
void LineMatching::onSimEventV2Output(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV2Output>& e)
{
  itsLines = e->getLines();

  if (SeC<SimEventInputFrame> eframe = q.check<SimEventInputFrame>(this)) 
  {
    GenericFrame frame = eframe->frame();
    itsInImage = frame.asRgb();
  }

  evolve(q);

  q.post(rutz::make_shared(new SimEventLineMatchingOutput(this, itsShapes)));
  
}

void LineMatching::evolve(SimEventQueue& q)
{
  LMLineMatcher lm;
	lm.Configure("para_line_matcher.txt");
	lm.Init("template_applelogo.txt");
	lm.GetColorImage(""); //THe image to display

  LFLineSegment* lines = new LFLineSegment[itsLines.size()];
	for (uint i=0 ; i<itsLines.size() ; i++)
  {
		lines[i].sx_ = itsLines[i].p1.i;
		lines[i].sy_ = itsLines[i].p1.j;
		lines[i].ex_ = itsLines[i].p2.i;
		lines[i].ey_ = itsLines[i].p2.j;
  }
  
  std::vector<Point2D<int> > maxPoly;
  if (0)
  {
   // std::vector<LMLineMatcher::Rect> matches = lm.Match(itsInImage.getWidth(), itsInImage.getHeight(), itsLines.size(), lines);
   // itsShapes.clear();
   // for(uint i=0; i<matches.size(); i++)
   //   itsShapes.push_back(Shape2D("applelogos", 
   //         matches[i].distance,
   //         Rectangle(Point2D<int>(matches[i].x, matches[i].y),
   //           Dims(matches[i].width, matches[i].height))));
  } else {

    lm.computeIDT3(itsInImage.getWidth(),
        itsInImage.getHeight(),
        itsLines.size(), lines);


    double  scaleParam = 0.5;
    EIEdgeImage* dbImages_ = new EIEdgeImage [1];
    dbImages_[0].SetNumDirections(60);
    dbImages_[0].Read((char*)"handdrawnApplelogoTemplate.txt");		
    dbImages_[0].Scale(scaleParam*1.0);


    LFLineSegment dbLine, queryLine;


    double ltrans[2];

    double factor = 1.0;
    double cost, mincost = 0;
    double minx, miny, maxx, maxy;
    double scale;

    LFLineSegment line;

    itsShapes.clear();
    for(double s = -1  ; s< 18 ; s++)	
    {		
      scale =  pow(1.1,s);
      printf("Scale %f\n", scale);
      EIEdgeImage tdbImage;
      tdbImage = dbImages_[0];
      tdbImage.Scale(scale);
      factor = 1.0/dbImages_[0].Length();
      tdbImage.Boundary(minx, miny, maxx, maxy);
      tdbImage.SetDirectionIndices();

      int width = itsInImage.getWidth()*scaleParam;
      int height = itsInImage.getHeight()*scaleParam;
     
      Image<PixRGB<byte> > tmp = itsInImage;
      for (int x=-(int)minx ; x<width-(int)minx ; x += 4)
      {
        for (int y=-(int)miny; y<height-(int)miny; y += 4)
        {

          ltrans[0] = (double)x;
          ltrans[1] = (double)y;				
          cost = 0;	

          if (minx + ltrans[0] <=4 ||
              minx + ltrans[0] >=width-4 || 
              maxx + ltrans[0] <=4 ||
              maxx + ltrans[0] >=width-4 ||
              miny + ltrans[1] <=4 ||
              miny + ltrans[1] >=height-4 || 
              maxy + ltrans[1] <=4 ||
              maxy + ltrans[1] >=height-4 )
          {
            cost = 1e+10;
            continue;
          }
          else
          {

            int count;
            cost = lm.getCost(tdbImage, ltrans, factor, count);
            std::vector<Point2D<int> > polygon;

            int pixCount = 0;
            for (int k=0 ; k<tdbImage.nLines_ ; k++)
            {
              LFLineSegment line = tdbImage.lines_[k];
              line.Translate(ltrans);
              polygon.push_back(Point2D<int>((int)line.sx_/scaleParam,(int)line.sy_/scaleParam));
              polygon.push_back(Point2D<int>((int)line.ex_/scaleParam,(int)line.ey_/scaleParam));
              pixCount += sqrt( squareOf(line.ey_-line.sy_) +
                                squareOf(line.ex_-line.sx_));
            }
            double prob = exp(-cost/ double(pixCount*3));
            //LINFO("Draw Cost %f %i %f \n", cost, pixCount, prob);
            //drawOutlinedPolygon(tmp, polygon, PixRGB<byte>(0,255,0));
            //SHOWIMG(tmp);

            


            if (prob>mincost) //mincost*1.0)
            {
              LINFO("Cost %i %i %f", x, y, prob);
              int x = (int)ceil((int)ltrans[0]/scaleParam-0.5);
              int y = (int)ceil((int)ltrans[1]/scaleParam-0.5);
              int detWindWidth= (int)(dbImages_[0].width_*scale/scaleParam);
              int detWindHeight= (int)(dbImages_[0].height_*scale/scaleParam);

              itsShapes.push_back(Shape2D("applelogos", 
                    cost,
                    Rectangle(Point2D<int>(x,y),
                      Dims(detWindWidth, detWindHeight)),
                    polygon));

              //if (cost < mincost)
              {
                mincost = prob;
                maxPoly = polygon;
              }

            }


          }
        }

      }

    }
    //LINFO("Cost %f\n", mincost);
    //drawOutlinedPolygon(itsInImage, maxPoly, PixRGB<byte>(0,255,0));
    //SHOWIMG(itsInImage);
  }


  //Output results
  for(uint i=0; i<itsShapes.size(); i++)
  {
    printf("Result: %i -1 -1 %i %i %i %i %f 0 0\n",
        i,
        itsShapes[i].bb.topLeft().i,
        itsShapes[i].bb.topLeft().j,
        itsShapes[i].bb.bottomRight().i,
        itsShapes[i].bb.bottomRight().j,
        itsShapes[i].score);
  }

}
  

// ######################################################################
void LineMatching::onSimEventSaveOutput(SimEventQueue& q,
    rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      if (disp.initialized())
        ofs->writeRgbLayout(disp, "LineMatching", FrameInfo("LineMatching", SRC_POS));
    }
}

Layout<PixRGB<byte> > LineMatching::getDebugImage()
{

  Layout<PixRGB<byte> > disp;

  for(uint i=0; i<itsShapes.size() && i < 4; i++)
  {
    if (itsInImage.rectangleOk(itsShapes[i].bb))
    {
      drawRect(itsInImage, itsShapes[i].bb, PixRGB<byte>(0,255,0));
      for(uint j=0; j<itsShapes[j].polygon.size()-1; j++)
      {
        drawLine(itsInImage, itsShapes[i].polygon[j],
            itsShapes[i].polygon[j+1], PixRGB<byte>(0,255,0));
      }
    }
  }
  disp = itsInImage;

  usleep(10000);
  return disp;
}

#endif
