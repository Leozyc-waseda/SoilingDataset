/*!@file SceneUnderstanding/Contours.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Contours.C $
// $Id: Contours.C 13906 2010-09-10 01:08:31Z lior $
//

#ifndef Contours_C_DEFINED
#define Contours_C_DEFINED


#include "Image/OpenCVUtil.H"
#include "plugins/SceneUnderstanding/LFLineFitter/LFLineFitter.h"
#include "plugins/SceneUnderstanding/Contours.H"
#include "plugins/SceneUnderstanding/V1.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/ColorMap.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/Convolutions.H"
#include "Simulation/SimEventQueue.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_Contours = {
  MOC_SORTPRI_3,   "Contours-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_ContoursShowDebug =
  { MODOPT_ARG(bool), "ContoursShowDebug", &MOC_Contours, OPTEXP_CORE,
    "Show debug img",
    "contours-debug", '\0', "<true|false>", "false" };


//Define the inst function name
SIMMODULEINSTFUNC(Contours);

// ######################################################################
Contours::Contours(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV2Output),
  SIMCALLBACK_INIT(SimEventContoursBias),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_ContoursShowDebug, this)
{

  itsCurrentContour = -1; //for debug view
}

// ######################################################################
Contours::~Contours()
{
}

// ######################################################################
void Contours::onSimEventV2Output(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV2Output>& e)
{
  itsTensorFields = e->getTensorFields();
  evolve(q);



  //Layout<PixRGB<byte> > layout = getDebugImage();
  //Image<PixRGB<byte> > tmp = layout.render();
  //SHOWIMG(tmp);
  
}

// ######################################################################
void Contours::onSimEventContoursBias(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventContoursBias>& e)
{
  itsBiasContours.clear();
  itsBiasContours = e->getContours();


  LINFO("Size %i", (uint)itsBiasContours.size());
  //Bias V1

  if (itsBiasContours.size() > 0)
  {
    std::vector<V1::SpatialBias> v1Bias;
    for(uint i=0; i<itsBiasContours.size(); i++)
    {
      Point2D<int> center(0,0);
      for(uint j=0; j<itsBiasContours[i].points.size(); j++)
      {
       // tmp.setVal(itsBiasContours[i].points[j], 255.0F);

        center += itsBiasContours[i].points[j];
      }
      center /= itsBiasContours[i].size();

      //Image<float> tmp(320,240,ZEROS);
      //drawCircle(tmp, center, 11, 255.0F);
      //SHOWIMG(tmp);

      //Bias V1
      V1::SpatialBias sb;
      sb.loc = center;
      sb.threshold = 0.01;
      sb.dims = Dims(50,50);
      v1Bias.push_back(sb);

    }
    q.post(rutz::make_shared(new SimEventV1Bias(this, v1Bias)));
    
  }

}


// ######################################################################
void Contours::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      if (disp.initialized())
        ofs->writeRgbLayout(disp, "Contours", FrameInfo("Contours", SRC_POS));
    }
}

// ######################################################################
void Contours::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (strcmp(e->getWinName(), "Contours"))
    return;
 
  switch(e->getKey())
  {
    case 10: //1
      itsCurrentContour++;
      if (itsCurrentContour > (int)itsContours.size()) itsCurrentContour = itsContours.size()-1;
      break;
    case 24: //q
      itsCurrentContour--;
      if (itsCurrentContour < 0) itsCurrentContour = 0;
      break;
    case 38: //a show all contours
      itsCurrentContour=-1;
      break;
  }
  

  evolve(q);

}

// ######################################################################
void Contours::evolve(SimEventQueue& q)
{



  //Process any biasing

  int cid=0; 
  itsContours.clear();
  //if (itsBiasContours.size() > 0)
  //{
  //  for(uint i=0; i<itsBiasContours.size(); i++)
  //  {
  //    for(uint j=0; j<itsBiasContours[i].points.size(); j++)
  //    {
  //      contoursImg.setVal(itsBiasContours[i].points[j], cid);
  //      itsBiasContours[i].ori.push_back(edgesOri.getVal(itsBiasContours[i].points[j]));
  //    }
  //    itsContours.push_back(itsBiasContours[i]);
  //    cid++;
  //  }
  //}



  Image<float> edgesMag; //the edges mag map
  Image<float> edgesOri; //the edges ori map
  for(uint cidx=0; cidx<3; cidx++)
  {
    if (!edgesMag.initialized())
    {
      edgesMag = Image<float>(itsTensorFields[cidx].getDims(), ZEROS);
      edgesOri = Image<float>(itsTensorFields[cidx].getDims(), ZEROS);
    }

    EigenSpace eigen = getTensorEigen(itsTensorFields[cidx].getTensorField());
    Image<float> features = eigen.l1-eigen.l2;
    for(uint i=0; i<features.size(); i++)
      if (features.getVal(i) > edgesMag.getVal(i))
      {
        edgesMag.setVal(i, features.getVal(i));
        float u = eigen.e1[1].getVal(i);
        float v = eigen.e1[0].getVal(i);
        edgesOri.setVal(i,atan(-u/v));
      }
  }
  //SHOWIMG(edgesMag);

  Image<float> contoursImg; //mask for which contours exsists
  std::vector<Point2D<int> > pointList = getPointList(edgesMag, edgesOri, contoursImg);

  //Go through the points and follow the contour
  for(uint i=0; i<pointList.size(); i++)
  {
    if (contoursImg.getVal(pointList[i]) == UNKNOWN) //since we dont know were this pixel belongs too lets follow it
    {
      Contour contour = followContour(cid, pointList[i], contoursImg, edgesMag, edgesOri);
      if (contour.points.size() > 5)
      {
        itsContours.push_back(contour);
        cid++;
      }
    }
  }


  //sort the contours
  std::sort(itsContours.begin(), itsContours.end(), ContourCmp());

  //////**** Debug ****////
  Image<PixRGB<byte> > byEdges = toRGB(itsTensorFields[2].getTokensMag(true));
  Image<PixRGB<byte> > tmp(byEdges.getDims(), ZEROS);

  for(uint i=0; i<itsContours.size(); i++)
  {
    Contour& contour = itsContours[i];
    for(uint j=0; j<contour.points.size(); j++)
      tmp.setVal(contour.points[j], PixRGB<byte>(0,255,0));
  }
  //SHOWIMG(tmp);




  q.post(rutz::make_shared(new SimEventContoursOutput(this, itsContours, edgesMag)));
}

Contours::Contour Contours::followContour(int idx, Point2D<int> startLoc, Image<float>& contoursImg,
    const Image<float>& edgesMag, const Image<float>& edgesOri)
{
  float origEdgeMag = edgesMag.getVal(startLoc);


  Image<float> t = edgesMag;
  inplaceNormalize(t, 0.0F, 255.0F);
  Image<PixRGB<byte> > tmp = t;


  Contour fcontour;
  fcontour.points.push_back(startLoc);
  fcontour.ori.push_back(edgesOri.getVal(startLoc));
  contoursImg.setVal(startLoc, idx);

  Contour contour;

  //follow the contour one way
  Point2D<int> pEdge(-1,-1);
  for(uint i=0; i<fcontour.points.size(); i++)
  {
    Point2D<int> newEdge = findEdgeToFollow(fcontour.points[i], origEdgeMag,
        edgesMag, edgesOri, contoursImg, pEdge);

    if (newEdge.isValid())
    {
      pEdge = newEdge;
      tmp.setVal(newEdge, PixRGB<byte>(0,255,0));
      contoursImg.setVal(newEdge,idx);
      fcontour.points.push_back(newEdge);
      fcontour.ori.push_back(edgesOri.getVal(newEdge));
    }
  }

  //Try to follow the other side of the contour
  Point2D<int> newEdge = findEdgeToFollow(startLoc, origEdgeMag,
      edgesMag, edgesOri, contoursImg, pEdge);

  if (newEdge.isValid()) //If we found this, then follow
  {
    tmp.setVal(newEdge, PixRGB<byte>(0,255,0));

    Contour bcontour;
    bcontour.points.push_back(newEdge);
    bcontour.ori.push_back(edgesOri.getVal(newEdge));
    contoursImg.setVal(newEdge,idx);

    for(uint i=0; i<bcontour.points.size(); i++)
    {
      Point2D<int> newEdge = findEdgeToFollow(bcontour.points[i], origEdgeMag,
          edgesMag, edgesOri, contoursImg, pEdge);

      if (newEdge.isValid())
      {
        tmp.setVal(newEdge, PixRGB<byte>(0,255,0));
        contoursImg.setVal(newEdge,idx);
        bcontour.points.push_back(newEdge);
        bcontour.ori.push_back(edgesOri.getVal(newEdge));
      }
      //Image<PixRGB<byte> > tt = rescale(tmp, tmp.getDims()*3); 
      //SHOWIMG(tt);
    }

    //combine the contours in order so the contour will be continues
    for(int i=bcontour.size()-1; i>=0; i--)
    {
      contour.points.push_back(bcontour.points[i]);
      contour.ori.push_back(bcontour.ori[i]);
    }

  }

  //combine the contours in order so the contour will be continues
  for(uint i=0; i<fcontour.size(); i++)
  {
    contour.points.push_back(fcontour.points[i]);
    contour.ori.push_back(fcontour.ori[i]);
  }

  return contour;
}

Point2D<int> Contours::findEdgeToFollow(const Point2D<int>& edgeLoc,
    const float origEdgeMag,
    const Image<float>& edgesMag, const Image<float>& edgesOri,
    const Image<float>& contoursImg,
    const Point2D<int>& pEdge)
{
  const int radius = 1; //search within this radius
  const float magDiffThreshold = 0.2;

  //float edgeMag = edgesMag.getVal(edgeLoc);
  //LINFO("Find edge Loc %ix%i", edgeLoc.i, edgeLoc.j);

  Point2D<int> newEdge(-1,-1);

  //Weights for biasing
  float w1 = 1;
  float w2 = 1;
  float w3 = 1.5;

  float maxDiff = 0;
  //Search for a new edge to add
  //LINFO("Getting edge to follow\n");
  for(int x=edgeLoc.i-radius; x<=edgeLoc.i+radius; x++)
    for(int y=edgeLoc.j-radius; y<=edgeLoc.j+radius; y++)
    {
      Point2D<int> loc(x,y);
      if (edgeLoc != loc &&
          contoursImg.coordsOk(loc) &&
          contoursImg.getVal(loc) == UNKNOWN)
      {
        float mag = 1.0F - (fabs(origEdgeMag - edgesMag.getVal(loc))/origEdgeMag);

        if (mag > magDiffThreshold) 
        {
          float edgeOri = edgesOri.getVal(edgeLoc);

          float dist = 1.0F - (edgeLoc.squdist(loc)/(2*radius*2));

          //Limit the search to only 90 degrees from us
          float oriDiff = 1.0F - (angDiff(edgeOri, edgesOri.getVal(loc))/(M_PI));

          //Point2D<int> t1 = pEdge - edgeLoc;
          //Point2D<int> t2 = loc - edgeLoc;
          //double edgeAng = angDiff(atan2(double(t1.j), double(t1.i)), atan2(double(t2.j), double(t2.i)));

          float diff = w1*mag + w2*dist + w3*oriDiff;
          //LINFO("m %f d %f o(%f,%f) %f eo: %f total: %f",
          //    mag, dist, edgeOri*180/M_PI, edgesOri.getVal(loc)*180/M_PI, oriDiff, edgeAng*180/M_PI, diff);
          if (diff > maxDiff)
          {
            newEdge = loc;
            maxDiff = diff;
          } 
        }
      }
    }
  //LINFO("Choose %f", maxDiff);

  return newEdge;
}



std::vector<Point2D<int> > Contours::getPointList(TensorVoting& tv, Image<float>& contoursImg,
    Image<float>& edgesMag, Image<float>& edgesOri)
{
  float maxMag = 1000;
  int nBins = 100;
  //float threshold = 0.1;

  //initialize the contoursImg; 
  contoursImg = Image<float>(tv.getDims(), NO_INIT);
  
  EigenSpace eigen = getTensorEigen(tv.getTensorField());
  Image<float> features = eigen.l1-eigen.l2;
  inplaceNormalize(features, 0.0F, maxMag);

  edgesMag = features;
  //SHOWIMG(edgesMag);
  edgesOri = Image<float>(features.getDims(), ZEROS);

  //Seperate the location above threshold into bins so we can sort them fast 
  //Mark the locations that we will be used in the contours img
  //
  std::vector<std::vector<Point2D<int> > > pointBins(nBins);
  for(int j=0; j<features.getHeight(); j++)
    for(int i=0; i<features.getWidth(); i++)
    {
      float val = features.getVal(i,j);
      if (val > 0) //threshold*maxMag)
      {
        contoursImg.setVal(i,j, UNKNOWN);
        //Find the bin this val belongs to
        int idx = (int)(val * nBins / maxMag);
        if (idx >= nBins) idx = nBins - 1; //if val == maxMag;
        pointBins[idx].push_back(Point2D<int>(i,j));

        //Get the direction of the vote from e1, while the weight is l1-l2
        float u = eigen.e1[1].getVal(i,j);
        float v = eigen.e1[0].getVal(i,j);
        edgesOri.setVal(i,j, atan(-u/v));
      } else {
        contoursImg.setVal(i,j, NOTDEFINED);
      }
    }

  //return a sorted list;
  std::vector<Point2D<int> > points;
  for(int i=pointBins.size()-1; i > -1; i--)
  {
    for(uint j=0; j<pointBins[i].size(); j++)
      points.push_back(pointBins[i][j]);
  }

  return points;

}

std::vector<Point2D<int> > Contours::getPointList(Image<float>& inMag, Image<float>& inOri, Image<float>& contoursImg)
{
  float maxMag = 1000;
  int nBins = 100;
  float threshold = 0.1;

  //initialize the contoursImg; 
  contoursImg = Image<float>(inMag.getDims(), NO_INIT);
  inplaceNormalize(inMag, 0.0F, maxMag);

  //Seperate the location above threshold into bins so we can sort them fast 
  //Mark the locations that we will be used in the contours img
  //
  std::vector<std::vector<Point2D<int> > > pointBins(nBins);
  for(int j=0; j<inMag.getHeight(); j++)
    for(int i=0; i<inMag.getWidth(); i++)
    {
      float val = inMag.getVal(i,j);
      if (val > threshold*maxMag)
      {
        contoursImg.setVal(i,j, UNKNOWN);
        //Find the bin this val belongs to
        int idx = (int)(val * nBins / maxMag);
        if (idx >= nBins) idx = nBins - 1; //if val == maxMag;
        pointBins[idx].push_back(Point2D<int>(i,j));
      } else {
        contoursImg.setVal(i,j, NOTDEFINED);
      }
    }

  //return a sorted list;
  std::vector<Point2D<int> > points;
  for(int i=pointBins.size()-1; i > -1; i--)
  {
    for(uint j=0; j<pointBins[i].size(); j++)
      points.push_back(pointBins[i][j]);
  }

  return points;

}


Layout<PixRGB<byte> > Contours::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;


  Layout<PixRGB<byte> > tensorDisp;
  Image<PixRGB<byte> > lumEdges = toRGB(itsTensorFields[0].getTokensMag(true));
  Image<PixRGB<byte> > rgEdges = toRGB(itsTensorFields[1].getTokensMag(true));
  Image<PixRGB<byte> > byEdges = toRGB(itsTensorFields[2].getTokensMag(true));

  tensorDisp = hcat(lumEdges, rgEdges);
  tensorDisp = hcat(tensorDisp, byEdges);


  //ShowContour;
  Image<PixRGB<byte> > contourImg(lumEdges.getDims(), ZEROS);
  ColorMap cm = ColorMap::LINES(100);
  if (itsCurrentContour != -1)
  {
      Contour& contour = itsContours[itsCurrentContour];
      for(uint j=0; j<contour.points.size(); j++)
        contourImg.setVal(contour.points[j], cm[itsCurrentContour%cm.size()]);

      char msg[255];
      sprintf(msg, "C: %i", itsCurrentContour);
      writeText(contourImg, Point2D<int>(0,0), msg,
          PixRGB<byte>(255,255,255),
          PixRGB<byte>(0,0,0));
      
  } else {
    for(uint i=0; i<itsContours.size(); i++)
    {
      Contour& contour = itsContours[i];
      for(uint j=0; j<contour.points.size(); j++)
        contourImg.setVal(contour.points[j], cm[i%cm.size()]);
    }
  }
  //for(uint i=0; i<itsBiasContours.size(); i++)
  //{
  //  for(uint j=0; j<itsBiasContours[i].points.size(); j++)
  //    contourImg.setVal(itsBiasContours[i].points[j], PixRGB<byte>(0,255,0));
  //}

  outDisp = contourImg; //vcat(tensorDisp, contourImg);
  
  return outDisp;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

