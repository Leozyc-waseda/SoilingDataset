/*!@file AppMedia/test-viewport3D.C test the opengl viewport */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-viewport3D.C $
// $Id: test-viewport3D.C 13054 2010-03-26 00:12:36Z lior $


#include "GUI/ViewPort3D.H"
#include "Util/log.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"

//start with 
//./bin/test-gpsView --out=display --in=/lab/tmp1id/u/NeovisionData/Videos/4.12.10_GPS_Test/PNMs/00134/stream-output#6#.pnm --input-frames=25-30-100000@33 1 3
//./bin/test-gpsView --out=display --in=/lab/tmp1id/u/NeovisionData/Videos/4.12.10_GPS_Test/PNMs/00135/stream-output#6#.pnm --input-frames=20-30-100000@33 1 699
//

struct GPSPos
{
  float latDeg;
  float latMin;
  float latSec;

  float lonDeg;
  float lonMin;
  float lonSec;

  GPSPos(float ld, float lm, float ls, float nd, float nm,  float ns) :
  latDeg(ld), latMin(lm), latSec(ls), lonDeg(nd), lonMin(nm), lonSec(ns)
  {}

  double getLat()
  {
      return  (latDeg + (latMin * 60.0) / 3600.0)*M_PI/180;
  }

  double getLon()
  {
      return (lonDeg + (lonMin * 60.0) / 3600.0)*M_PI/180;
  }
    

};

Point3D<float>  getGpsPos(GPSPos pos)
{
  GPSPos center(34,1.270,0,118,17.334,0);

  Point3D<float> relPos(0,0,0);

  //Find Y movement
  double R = 6378.7 ; //Earth Radians (KM)
  double theta = center.getLon() - pos.getLon();
  double d1 = sin(center.getLat()) * sin(center.getLat());
  double d2 = cos(center.getLat()) * cos(center.getLat()) * cos(theta);
  double y = acos(d1 + d2) * R * 1000.0;//meter

  LINFO("%f %f %f %f %f", R, theta, d1, d2, y);
  //Find X movement
  d1 = sin(center.getLat()) * sin(pos.getLat());
  d2 = cos(center.getLat()) * cos(pos.getLat());
  double x = acos(d1 + d2) * R * 1000.0;//meter

  if(x != 0.0 && y != 0.0)
    {
      relPos.x = (center.getLat() > pos.getLat()) ? x : -x;
      relPos.y = (center.getLon() > pos.getLon()) ? y : -y;
    }

  return relPos;
}


Point2D<float>  getGpsPos(double lon1, double lat1, double lon2, double lat2)
{
  Point2D<float> relPos(0,0);

  //Find Y movement
  double R = 6378.7 ; //Earth Radians (KM)
  double theta = lon1- lon2;
  double d1 = sin(lat1) * sin(lat1);
  double d2 = cos(lat1) * cos(lat1) * cos(theta);

  double y = 0;
  if (-1 <= d1 + d2 &&
      1 >= d1 + d2)
    y = acos(d1 + d2) * R * 1000.0;//meter

  //Find X movement
  d1 = sin(lat1) * sin(lat2);
  d2 = cos(lat1) * cos(lat2);

  double x = 0;
  if (-1 <= d1 + d2 &&
      1 >= d1 + d2)
    x = acos(d1 + d2) * R * 1000.0;//meter

  //if(x != 0.0 && y != 0.0)
    {
      relPos.i = (lat1 > lat2) ? x : -x;
      relPos.j = (lon1 > lon2) ? y : -y;
    }

  return relPos;
}

Image<float> computeAffine()
{

  std::vector<Point2D<double> > gpsLocations;
  //gpsLocations.push_back(Point2D<float>(34.02172348,-118.29089056)); 
  //gpsLocations.push_back(Point2D<float>(34.02186136,-118.29089056));
  //gpsLocations.push_back(Point2D<float>(34.02194239,-118.29116427));
  //gpsLocations.push_back(Point2D<float>(34.02185214,-118.29101424));
  //gpsLocations.push_back(Point2D<float>(34.02180715,-118.29088697));

  gpsLocations.push_back(Point2D<double>(34.021790332,-118.290910238)); 
  gpsLocations.push_back(Point2D<double>(34.021877944,-118.290840708));
  gpsLocations.push_back(Point2D<double>(34.021991116,-118.291105977));
  gpsLocations.push_back(Point2D<double>(34.021938097,-118.291058661));
  gpsLocations.push_back(Point2D<double>(34.021864139,-118.290929807));



  std::vector<Point2D<double> > gtLocations;
  gtLocations.push_back(Point2D<double>(0,0));
  gtLocations.push_back(Point2D<double>(14.5796,0));
  gtLocations.push_back(Point2D<double>(14.5796,27.9654));
  gtLocations.push_back(Point2D<double>(8.4582,1.8542+11.91));
  gtLocations.push_back(Point2D<double>(8.4582,1.8542));

  printf("D=[ \n");
  for(uint i=0; i<gpsLocations.size(); i++)
  {
    Point2D<float> pos = getGpsPos(
        double(-118*M_PI/180), double(34*M_PI/180),
        gpsLocations[i].j*M_PI/180, gpsLocations[i].i*M_PI/180);

    printf("%f %f %f %f; \n", 
        pos.i, pos.j, gtLocations[i].i, gtLocations[i].j);

  }
  printf("];\n");

  printf("gps=[D(:,1)'; D(:,2)'; ones(1,size(D,1))]; \n");
  printf("gt=[D(:,3)'; D(:,4)'; ones(1,size(D,1))]; \n");
  printf("M = gt / gps\n");

  /*
  // we are going to solve the linear system Ax=b in the least-squares sense
  Image<float> A(3, nPoints, NO_INIT);
  Image<float> b(3, nPoints, NO_INIT);

  for (int i = 0; i < nPoints; i ++)
    {
      Point2D<float> pos = getGpsPos(
          double(-118*M_PI/180), double(34*M_PI/180),
          locData[i][1]*M_PI/180, locData[i][0]*M_PI/180);
      LINFO("LocData %f %f", pos.i, pos.j);

      A.setVal(0, i, pos.i);
      A.setVal(1, i, pos.j);
      A.setVal(2, i, 1.0f);

      b.setVal(0, i, locData[i][2]);
      b.setVal(1, i, locData[i][3]);
      b.setVal(2, i, 1.0f);
    }

  Image<float> aff;
  try
    {
      // the solution to Ax=b is x = [A^t A]^-1 A^t b:
      Image<float> At = transpose(A);

      Image<float> x =
        matrixMult(matrixMult(matrixInv(matrixMult(At, A)), At), b);

      aff = x;
    } 
  catch (SingularMatrixException& e)
    {
      LINFO("Couldn't invert matrix -- RETURNING IDENTITY");
    }

  return aff;
  */
  
  Image<float> aff;
  return aff;
}


void showPos(ViewPort3D& vp, GPSPos pos)
{
  Point3D<float> relPos = getGpsPos(pos); 
  //Test the sphere display
  vp.drawBox( relPos, //Position
      Point3D<float>(0,0,0), //Rotation
      Point3D<float>(0.52324,0.5/4, 1.7), //size
      PixRGB<byte>(256,256,256)
      );
}

Image<PixRGB<byte> > getScene(ViewPort3D& vp, double gtx, double gty)
{
  vp.initFrame();

  //Draw Ground truth
  vp.drawBox( Point3D<float>(0,0,0), //Position
      Point3D<float>(0,0,0), //Rotation
      Point3D<float>(9.14400, 9.14400, 0.001), //size
      PixRGB<byte>(256,256,256)
      );

  vp.drawBox( Point3D<float>(-(10.9728/2)+(9.144/2),-(23.7744/2)+(9.144/2),0), //Position
      Point3D<float>(0,0,0), //Rotation
      Point3D<float>(10.9728, 23.7744, 0.001), //size
      PixRGB<byte>(256,256,256)
      );

  if (gtx != -1)
  {
    vp.drawBox( Point3D<float>(gtx-(9.144/2),
          gty-(9.144/2),1.7/2), //Position
        Point3D<float>(0,0,0), //Rotation
        Point3D<float>(0.52324,0.5/4, 1.7), //size of a average human
        PixRGB<byte>(256,256,256)
        );
  }

  Image<PixRGB<byte> > img(vp.getDims(), ZEROS);
  std::vector<ViewPort3D::Line> lines = vp.getFrameLines();
  for(uint i=0; i<lines.size(); i++)
    drawLine(img, Point2D<int>(lines[i].p1), Point2D<int>(lines[i].p2), PixRGB<byte>(0,255,0));


  return img;


}


Image<PixRGB<byte> > getPersonCenter(ViewPort3D& vp, Image<float>& aff,
    double px, double py, Point2D<float>& center)
{
  vp.initFrame();

  //Draw the location of the person
  Point2D<float> pos = getGpsPos(
      double(-118*M_PI/180), double(34*M_PI/180),
      py*M_PI/180, px*M_PI/180);

  Image<float> x(3, 1, NO_INIT);
  x.setVal(0, 0, pos.i);
  x.setVal(1, 0, pos.j);
  x.setVal(2, 0, 1.0f);


  Image<float> mappedLoc = matrixMult(x,aff);

  vp.drawBox( Point3D<float>(mappedLoc.getVal(0,0), mappedLoc.getVal(1,0),1.7/2), //Position
      Point3D<float>(0,0,0), //Rotation
      Point3D<float>(0.52324,0.5/4, 1.7), //size of a average human
      PixRGB<byte>(256,256,256)
      );

  center = Point2D<float>(0,0);
  Image<PixRGB<byte> > img(vp.getDims(), ZEROS);
  std::vector<ViewPort3D::Line> lines = vp.getFrameLines();
  for(uint i=0; i<lines.size(); i++)
  {
    drawLine(img, Point2D<int>(lines[i].p1), Point2D<int>(lines[i].p2), PixRGB<byte>(0,0,255));

    center += lines[i].p1;
    center += lines[i].p2;
  }
  center /= lines.size()*2;

  return img;

}

int getKey(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("gpsOutput")
    : rutz::shared_ptr<XWinManaged>();
  if (uiwin.is_valid())
    return uiwin->getLastKeyPress();
  else
    return -1;
}

Point2D<int> getMouseClick(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("ViewPort3D")
    : rutz::shared_ptr<XWinManaged>();

  if (uiwin.is_valid())
    return uiwin->getLastMouseClick();
  else
    return Point2D<int>(-1,-1);
}


std::vector<Point2D<double> > getGpsTracks(const char* filename)
{

  std::vector<Point2D<double> > gpsTracks;

  //test-gpsView::main: Frame 0 idx: 0 (34.021790,118.290910)

 gpsTracks.push_back(Point2D<double>(34.021790332,-118.290910238)); 
 gpsTracks.push_back(Point2D<double>(34.021877944,-118.290840708));
 gpsTracks.push_back(Point2D<double>(34.021991116,-118.291105977));
 gpsTracks.push_back(Point2D<double>(34.021938097,-118.291058661));
 gpsTracks.push_back(Point2D<double>(34.021864139,-118.290929807));

 //return gpsTracks;


  //FILE * pFile = fopen (filename,"r");
  //if (pFile == NULL)
  //  LFATAL("Cannot open gps file %s", filename);

  //while(!feof(pFile))
  //{
  //  double lat,lon;
  //  if(fscanf(pFile, "%lf %lf", &lat, &lon));;
  //  gpsTracks.push_back(Point2D<double>(lat,lon));
  //}
  //  
  //fclose (pFile);

  return gpsTracks;
}

int main(int argc, char *argv[]){

  ModelManager manager("Test Viewport");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);
  

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<fps> <start>", 2, 2) == false) return(1);
  // let's get all our ModelComponent instances started:
  manager.start();

  float fps = atof(manager.getExtraArg(0).c_str());
  int startFrame = atoi(manager.getExtraArg(1).c_str());
  //Normal operation
  //

  //Get the gps points from the file
  std::vector<Point2D<double> > gpsTracks = getGpsTracks("gpsTracks.txt");

  //computeAffine();
  //getchar();
  Image<float> aff(3,3,ZEROS);
  //aff.setVal(0,0, 5.606325242018370e-01); aff.setVal(0,1, -7.595550353425364e-01 ); aff.setVal(0,2, 2.145894761926837e+04 );
  //aff.setVal(1,0, -9.750888835556519e-01); aff.setVal(1,1, -5.651409544700519e-01); aff.setVal(1,2, 1.268057832192886e+04 );
  //aff.setVal(2,0, 2.588141873984341e-21 ); aff.setVal(2,1, -1.652135135951789e-20 ); aff.setVal(2,2, 1.0000e+00 );

  //aff.setVal(0,0,4.699159160488188e-01); aff.setVal(0,1, -8.732876848322884e-01 ); aff.setVal(0,2, 2.425961617339588e+04 );
  //aff.setVal(1,0, -9.485676912296747e-01); aff.setVal(1,1, -4.504131693298014e-01); aff.setVal(1,2, 9.702550452350131e+03 );
  //aff.setVal(2,0, -3.087034376690811e-21 ); aff.setVal(2,1, 5.309152659651456e-21 ); aff.setVal(2,2, 9.999999999999999e-01 );
  
  aff.setVal(0,0,-1.12208358608957); aff.setVal(0,1,  -0.622078400100041 ); aff.setVal(0,2, 13980.5274210865 );
  aff.setVal(1,0, -0.617532176829138); aff.setVal(1,1, 0.709870729691164); aff.setVal(1,2, -20560.1790593578 );
  aff.setVal(2,0, 0 ); aff.setVal(2,1, 0 ); aff.setVal(2,2, 1 );

  ViewPort3D vp(1920,1088, true, false, true);
  vp.setProjectionMatrix(27, 1.680, 0.005, 500);
  //Point3D<float> camPos(-3,-16.6,33.6);
  //Point3D<float> camOri(154,-4,-21);


  //Point3D<float> camPos(-18.000032, -50.499775, 30.900019);
  //Point3D<float> camOri(120.000000, -1.000000, -21.000000);
  
  Point3D<float> camPos(-7.7, -35.8, 20.0);
  Point3D<float> camOri(117.000000, 0.000000, -26.000000);

  //Point3D<float> camPos(0,0,0.32004);
  //Point3D<float> camOri(180,0,0);


  //float fps=20;
  Point2D<int> lastLoc(512/2,512/2);
  Point2D<int> clickPos(-1,-1);
  //camPos.x = 0; camPos.y=0;
  
  Image< PixRGB<byte> > inputImg;
  //ifs->updateNext();
  //GenericFrame input = ifs->readFrame();
  ////if (!input.initialized())
  ////  break;
  //inputImg = input.asRgb();
  int frameOne = ifs->frame();
  for(float frame=startFrame; frame<240000*fps; frame++)
  {
     ifs->updateNext();
    //const FrameState is = ifs->updateNext();
    //if (is == FRAME_COMPLETE)
    //  break;

    //grab the images
    GenericFrame input = ifs->readFrame();
    //if (!input.initialized())
    //  break;
    inputImg = input.asRgb();



    int key = getKey(ofs);
    switch(key)
    {
      case 113: camPos.x -= 0.1; break; //left
      case 114: camPos.x += 0.1; break; //right
      case 111: camPos.y -= 0.1; break; //up
      case 116: camPos.y += 0.1; break; //down
      case 38: camPos.z += 0.1; break; //a
      case 52: camPos.z -= 0.1; break; //z

      case 10: camOri.x += 1; break; //1
      case 24: camOri.x -= 1; break; //q
      case 11: camOri.y += 1; break; //2
      case 25: camOri.y -= 1; break; //w
      case 12: camOri.z += 1; break; //3
      case 26: camOri.z -= 1; break; //e

      default:
        if (key != -1)
          LINFO("Key %i", key);
        break;
    }
    vp.setCamera(camPos, camOri);

    Point2D<int> mouseLoc =  getMouseClick(ofs);
    if (mouseLoc.isValid())
    {
      clickPos = mouseLoc;

      Point3D<float> pos3D = vp.getPosition(Point3D<float>(clickPos.i, clickPos.j, -29));
      LINFO("Click Pos %ix%i %fx%fx%f", clickPos.i, clickPos.j, pos3D.x, pos3D.y, pos3D.z);
    }

    //LINFO("Pos %f, %f, %f Ori %f, %f, %f",
    //    camPos.x, camPos.y, camPos.z,
    //    camOri.x, camOri.y, camOri.z);
    
    int idx = (int(float(ifs->frame()-frameOne)/fps) + startFrame)%gpsTracks.size();

    LINFO("Frame %i idx: %i (%f,%f)", ifs->frame() - frameOne, idx, gpsTracks[idx].i, gpsTracks[idx].j );
    Point2D<float> personCenter;
    Image<PixRGB<byte> > projectedScene = getPersonCenter(vp, aff,
        gpsTracks[idx].i, gpsTracks[idx].j, personCenter);

    ////Draw Ground truth
    vp.initFrame();
    //vp.drawBox( Point3D<float>(0,0,0), //Position
    //    Point3D<float>(0,0,0), //Rotation
    //    Point3D<float>(7.6454,9.2964, 0.001), //size
    //    PixRGB<byte>(256,256,256)
    //    );

    vp.drawCircle(Point3D<float>(0,0,0), Point3D<float>(0,0,0), 0.25, PixRGB<byte>(255,0,0));
    vp.drawCircle(Point3D<float>(14.5796,0,0), Point3D<float>(0,0,0), 0.25, PixRGB<byte>(255,0,0));
    vp.drawCircle(Point3D<float>(14.5796,27.9654,0), Point3D<float>(0,0,0), 0.25, PixRGB<byte>(255,0,0));
    vp.drawCircle(Point3D<float>(8.4582,1.8542+11.91,0), Point3D<float>(0,0,0), 0.25, PixRGB<byte>(255,0,0));
    vp.drawCircle(Point3D<float>(8.4582,1.8542,0), Point3D<float>(0,0,0), 0.25, PixRGB<byte>(255,0,0));
    vp.drawGrid(Dims(100,100), Dims(5,5));

    //vp.drawBox( Point3D<float>(0, 0,2.58445/2), //Position
    //    Point3D<float>(0,0,0), //Rotation
    //    Point3D<float>(2.41935,10, 2.58445 ), //size of a average human
    //    //Point3D<float>(2,2, 2 ), //size of a average human
    //    PixRGB<byte>(256,256,256)
    //    );
    

    Image<PixRGB<byte> > img(vp.getDims(), ZEROS);
    std::vector<ViewPort3D::Line> lines = vp.getFrameLines();
    for(uint i=0; i<lines.size(); i++)
      drawLine(img, Point2D<int>(lines[i].p1), Point2D<int>(lines[i].p2), PixRGB<byte>(0,128,0),1);

    Image<PixRGB<byte> > tmp = inputImg + img;

    int cropWidth = 130;
    drawCircle(tmp, Point2D<int>(personCenter), cropWidth, PixRGB<byte>(255,0,0));

    if (inputImg.coordsOk(Point2D<int>((int)personCenter.i + cropWidth, (int)personCenter.j + cropWidth)) )
    {
      Point2D<int> tl((int)personCenter.i - cropWidth, (int)personCenter.j - cropWidth);

      if (inputImg.coordsOk(Point2D<int>((int)personCenter.i - cropWidth, (int)personCenter.j - cropWidth)))
      {
        Image<PixRGB<byte> > patch = crop(inputImg, tl, Dims(cropWidth*2, cropWidth*2));

        inplacePaste(tmp, patch, Point2D<int>(100,900-cropWidth*2));
        drawRect(tmp, Rectangle(Point2D<int>(100,900-cropWidth*2), Dims(cropWidth*2, cropWidth*2)),
            PixRGB<byte>(0,255,0),3);
      } else {
        if (tl.i <= 0) tl.i = 0;
        if (tl.j <= 0) tl.j = 0;

        Image<PixRGB<byte> > patch = crop(inputImg, tl, Dims(cropWidth*2, cropWidth*2));

        inplacePaste(tmp, patch, Point2D<int>(100,900-cropWidth*2));
        drawRect(tmp, Rectangle(Point2D<int>(100,900-cropWidth*2), Dims(cropWidth*2, cropWidth*2)),
            PixRGB<byte>(0,255,0),3);
      }

    }

   // inputImg = rescale(inputImg, 640*2, 480*2);
    ofs->writeRGB(tmp, "gpsOutput", FrameInfo("gpsOutput", SRC_POS));
    usleep(10000);
 //   getchar();
  }

  exit(0);

}
