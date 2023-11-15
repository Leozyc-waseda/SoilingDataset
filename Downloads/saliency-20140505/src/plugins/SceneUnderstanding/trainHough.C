/*! @file SceneUnderstanding/trainHough.C train the hough transform */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/trainHough.C $
// $Id: trainHough.C 13765 2010-08-06 18:56:17Z lior $
//

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Component/JobServerConfigurator.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"
#include "Image/MatrixOps.H"
#include "Image/Point2D.H"
#include "Media/SimFrameSeries.H"
#include "Neuro/NeuroOpts.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Simulation/SimEventQueue.H"
#include "Util/AllocAux.H"
#include "Util/Pause.H"
#include "Util/Types.H"
#include "Util/csignals.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "rutz/trace.h"
#include "plugins/SceneUnderstanding/Geons3D.H"
#include "plugins/SceneUnderstanding/SMap.H"
#include "plugins/SceneUnderstanding/CornersFeatures.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "GUI/DebugWin.H"
#include "GUI/ViewPort3D.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"

#include <signal.h>
#include <sys/types.h>

std::vector<CornersFeatures::CornerState> getCorners(std::vector<V2::LineSegment>& lines)
{
  std::vector<CornersFeatures::CornerState> corners;
  float minDist = 5;
  float minDistSq = minDist*minDist;

  //Image<float> edges(320, 240, ZEROS);
  //for(uint i=0; i<lines.size(); i++)
  //{
  //  V2::LineSegment& ls1 = itsLines[i];
  //  drawLine(edges, Point2D<int>(ls1.p1), Point2D<int>(ls1.p2), 255.0F);
  //}


  for(uint i=0; i<lines.size(); i++)
  {
    V2::LineSegment& ls1 = lines[i];

    //Image<PixRGB<byte> > tmp(320,240,ZEROS); // = edges;
    //drawLine(tmp, Point2D<int>(ls1.p1), Point2D<int>(ls1.p2), PixRGB<byte>(255,0,0));

    std::vector<float> angles1; //The angles from p1 prospective
    std::vector<float> angles2; //the angles from p2 prospective
    double dx = ls1.p2.i - ls1.p1.i;
    double dy = ls1.p2.j - ls1.p1.j;
    double ang = atan2(dx, dy) + M_PI/2;
    angles1.push_back(ang);
    angles2.push_back(ang+M_PI);

    for(uint j=i+1; j<lines.size(); j++)
    {
      if (i == j)
        continue;
      V2::LineSegment& ls2 = lines[j];

      //Find if line segment i p1 intesect line segment j, and find the angle betwwen this
      //point and the reset of the ends
      if (ls1.p1.distanceToSegment(ls2.p1, ls2.p2) < minDist)
      {
        double dx1 = ls2.p1.i - ls1.p1.i;
        double dy1 = ls2.p1.j - ls1.p1.j;

        double dx2 = ls2.p2.i - ls1.p1.i;
        double dy2 = ls2.p2.j - ls1.p1.j;

        //If we intresected on a line then add both ends
        if ( (dx1*dx1 + dy1*dy1) > minDistSq) //p1 is further
          angles1.push_back(atan2(dx1, dy1) + M_PI/2);

        if ( (dx2*dx2 + dy2*dy2) > minDistSq) //p2 is further
          angles1.push_back(atan2(dx2, dy2) + M_PI/2);
      }


      //Do the same for p2 in line segment i
      if (ls1.p2.distanceToSegment(ls2.p1, ls2.p2) < minDist)
      {
        double dx1 = ls2.p1.i - ls1.p2.i;
        double dy1 = ls2.p1.j - ls1.p2.j;

        double dx2 = ls2.p2.i - ls1.p2.i;
        double dy2 = ls2.p2.j - ls1.p2.j;

        //If we intresected on a line then add both ends
        if ( (dx1*dx1 + dy1*dy1) > minDistSq) //p1 is further
          angles2.push_back(atan2(dx1, dy1) + M_PI/2);

        if ( (dx2*dx2 + dy2*dy2) > minDistSq) //p2 is further
          angles2.push_back(atan2(dx2, dy2) + M_PI/2);
        
      }
    }

    //Add the two corners

    CornersFeatures::CornerState c1;
    c1.center = ls1.p1;
    c1.angles = angles1;
    corners.push_back(c1);

    CornersFeatures::CornerState c2;
    c2.center = ls1.p2;
    c2.angles = angles2;
    corners.push_back(c2);

  }
  //SHOWIMG(tmp);

  //for(uint i=0; i<corners.size(); i++)
  //{
  //  Image<PixRGB<byte> > tmp(320, 240, ZEROS);
  //  for(uint ai=0; ai<corners[i].angles.size(); ai++)
  //  {
  //    int x1 = int(cos(corners[i].angles[ai])*30.0/2.0);
  //    int y1 = int(sin(corners[i].angles[ai])*30.0/2.0);
  //    Point2D<float> p1(corners[i].center.i-x1, corners[i].center.j+y1);

  //    drawLine(tmp, Point2D<int>(corners[i].center), Point2D<int>(p1), PixRGB<byte>(0,255,0));
  //  }
  //  SHOWIMG(tmp);
  //}


  return corners;

}


std::vector<V2::LineSegment> getLineSegments(ViewPort3D& vp, 
    Point3D<float>& objPos, Point3D<float>& objRot, Geons3D::GeonType type)
{
  vp.setWireframeMode(false);
  vp.setLightsMode(true);
  vp.setFeedbackMode(false);

  vp.initFrame();

  switch(type)
  {
    case Geons3D::BOX:
      {
        vp.drawBox(
            objPos,
            objRot,
            Point3D<float>(30,30,30),
            PixRGB<byte>(0,256,0));
      }
      break;
    case Geons3D::SPHERE:
      break;
    case Geons3D::CYLINDER:
        vp.drawCylinder(
            objPos,
            objRot,
            15, 32.5,
            PixRGB<byte>(0,256,0));
      break;

  }

  Image<PixRGB<byte> > frame =  flipVertic(vp.getFrame());
  Image<float> surface = luminance(frame);

  Image<float> edgesMag, edgesOri;
  gradientSobel(surface, edgesMag, edgesOri);

  vp.setWireframeMode(true);
  vp.setLightsMode(false);
  vp.setFeedbackMode(true);

  vp.initFrame();

  switch(type)
  {
    case Geons3D::BOX:
      {
        vp.drawBox(
            objPos,
            objRot,
            Point3D<float>(30,30,30),
            PixRGB<byte>(0,256,0));
      }
      break;
    case Geons3D::SPHERE:
      break;
    case Geons3D::CYLINDER:
        vp.drawCylinder(
            objPos,
            objRot,
            15, 32.5,
            PixRGB<byte>(0,256,0));
      break;

  }

  std::vector<ViewPort3D::Line> lines = vp.getFrameLines();

  Image<float> mag(vp.getDims(),ZEROS);
  Image<float> ori(vp.getDims(),ZEROS);
  
  inplaceNormalize(edgesMag, 0.0F, 100.0F);
  std::vector<V2::LineSegment> lineSegments;
  for(uint i=0; i<lines.size(); i++)
  {

    double dx = lines[i].p2.i - lines[i].p1.i;
    double dy = lines[i].p2.j - lines[i].p1.j;
    double ang = atan2(dx,dy) + M_PI/2;
    //Change orientation from 0 to M_PI
    if (ang < 0) ang += M_PI;
    if (ang >= M_PI) ang -= M_PI;

    //Get the center of the line
    Point2D<float> center = lines[i].p1 + Point2D<float>(dx/2,dy/2); 

    if (edgesMag.coordsOk(Point2D<int>(lines[i].p1)) &&
        edgesMag.coordsOk(Point2D<int>(lines[i].p2)) &&
        edgesMag.getVal(Point2D<int>(lines[i].p1)) > 10 &&
        edgesMag.getVal(Point2D<int>(lines[i].p2)) > 10 &&
        edgesMag.getVal(Point2D<int>(center)) > 10)
    {
      double length = sqrt(dx*dx + dy*dy);

      V2::LineSegment ls;
      ls.p1 = lines[i].p1;
      ls.p2 = lines[i].p2;
      ls.length = length;
      ls.ori = ang;
      ls.center = center;

      lineSegments.push_back(ls);
    }
  }
  
  return lineSegments;

}

Image<float> getSmap(ViewPort3D& vp, std::vector<V2::LineSegment>& lines)
{

  Image<float> smap(vp.getDims(), ZEROS);

  if (lines.size() <= 0)
    return smap;
  
  //Calculate the bounding box and center
  Point2D<int> center(0,0);
  Point2D<int> tl(0,0);
  Point2D<int> br(vp.getDims().w(),vp.getDims().h());

  for(uint i=0; i<lines.size(); i++)
  {
    center.i += int(lines[i].center.i);
    center.j += int(lines[i].center.j);

    if (lines[i].p1 > tl)
      tl = (Point2D<int>)lines[i].p1;
    if (lines[i].p2 > tl)
      tl = (Point2D<int>)lines[i].p2;

    if (lines[i].p1 < br)
      tl = (Point2D<int>)lines[i].p1;
    if (lines[i].p2 > tl)
      tl = (Point2D<int>)lines[i].p2;

  //  drawLine(smap, (Point2D<int>)lines[i].p1, (Point2D<int>)lines[i].p2, 255.0F);
  }
  center /= lines.size();

  drawDisk(smap, center, 3, 255.0F);
 // drawCircle(smap, tl, 3, 255.0F);
 // drawCircle(smap, br, 3, 255.0F);

  return smap;

}



int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Train Hough");

  nub::ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<Geons3D> geons3D(new Geons3D(manager));
  manager.addSubComponent(geons3D);
  
  // Request a bunch of option aliases (shortcuts to lists of options):
  REQUEST_OPTIONALIAS_NEURO(manager);
  

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  nub::ref<SimEventQueue> seq = seqc->getQ();
  
  manager.start();

  GHough ghough;


  ViewPort3D vp(320,240, true, false, true);
  //double trans[3][4] = {
  //  {-0.996624, 0.070027, 0.042869, -16.907477},
  //  {-0.004359, 0.476245, -0.879302, 9.913470},
  //  {-0.081990, -0.876520, -0.474332, 276.648010}};

  double trans[3][4] = {
    {-0.999143, -0.041185, -0.004131, -1.142130},
    {-0.017002, 0.499358, -0.866229, 17.284269},
    {0.037739, -0.865416, -0.499630, 220.977236}};
  
  vp.setCamera(trans);

  Point3D<float> objPos(0,0,16/2);
  Point3D<float> objRot(0,0,0);


  // temporary, for debugging...
  seq->printCallbacks();

  PauseWaiter p;
  //int retval = 0;
  SimStatus status = SIM_CONTINUE;

  for(int obj=0; obj<3; obj++)
  for(float x=-150; x<150; x+=5)
    for(float y=40; y>-150; y-=5)
      for(float z=0; z<90; z+=5)
      {

        int objType = 0;

        switch (obj)
        {
          case 0:
            objType = 0;
            objPos.x = x;
            objPos.y = y;
            objRot.z = z;
            break;
          case 1:
            objType = 1;
            objPos.x = x;
            objPos.y = y;
            objRot.z = 0;
            z+= 90; //No rotation for object 1
            break;
          case 2:
            objType = 1;
            objPos.x = x;
            objPos.y = y;

            objRot.x = 90;
            objRot.y = z;
            objRot.z = 0;
            break;
        }

        //float x = 26.80;
        //float y = -44.90;
        //float z = 52;
        printf("%i %f %f %f\n",obj, x, y, z);


        std::vector<V2::LineSegment> lines = getLineSegments(vp, objPos, objRot, (Geons3D::GeonType)objType);

        std::vector<CornersFeatures::CornerState> corners = getCorners(lines);


        //for(uint i=0; i<corners.size(); i++)
        //{
        //  Image<PixRGB<byte> > tmp(320, 240, ZEROS);
        //  for(uint ai=0; ai<corners[i].angles.size(); ai++)
        //  {
        //    int x1 = int(cos(corners[i].angles[ai])*30.0/2.0);
        //    int y1 = int(sin(corners[i].angles[ai])*30.0/2.0);
        //    Point2D<float> p1(corners[i].center.i-x1, corners[i].center.j+y1);

        //    drawLine(tmp, Point2D<int>(corners[i].center), Point2D<int>(p1), PixRGB<byte>(0,255,0));
        //  }
        //  SHOWIMG(tmp);
        //}
        


        //Send an SMap
        Image<byte> smap = getSmap(vp, lines);
        std::vector<SMap::SMapState> smapState;
        seq->post(rutz::make_shared(new SimEventSMapOutput(geons3D.get(), smapState, smap)));

        //Send Corners
        seq->post(rutz::make_shared(new SimEventCornersOutput(geons3D.get(), corners)));

        //Send metadata
        //rutz::shared_ptr<GenericFrame::MetaData> metaData;
        rutz::shared_ptr<World3DInput::ObjectsData> objectsData(new World3DInput::ObjectsData);
        Point3D<float> color(1,1,1);
        Point3D<float> params(30,30,30);
        objectsData->objects.push_back(
            World3DInput::Object(World3DInput::Object::ObjectType(objType),
              objPos, objRot, color, params));
        ImageSet<float> cells;
        rutz::shared_ptr<GenericFrame::MetaData> metaData = objectsData;
        seq->post(rutz::make_shared(new SimEventLGNOutput(geons3D.get(), cells, metaData)));
        seq->post(rutz::make_shared(new SimEventV2Output(geons3D.get(), lines, vp.getDims())));

        // Evolve for one time step and switch to the next one:
        status = seq->evolve();
 
        Layout<PixRGB<byte> > disp = geons3D->getDebugImage(*seq);
        ofs->writeRgbLayout(disp, "Geons3D", FrameInfo("Geons3D", SRC_POS));
        

      }


  manager.stop();

  return 0;
}

