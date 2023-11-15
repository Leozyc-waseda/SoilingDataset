/*!@file SceneUnderstanding/LocalParts.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/LocalParts.C $
// $Id: LocalParts.C 13701 2010-07-27 01:08:12Z lior $
//

#ifndef LocalParts_C_DEFINED
#define LocalParts_C_DEFINED

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/Convolutions.H"
#include "Image/MatrixOps.H"
#include "Simulation/SimEventQueue.H"
#include "plugins/SceneUnderstanding/LocalParts.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_LocalParts = {
  MOC_SORTPRI_3,   "LocalParts-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_LocalPartsShowDebug =
  { MODOPT_ARG(bool), "LocalPartsShowDebug", &MOC_LocalParts, OPTEXP_CORE,
    "Show debug img",
    "localparts-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(LocalParts);


// ######################################################################
LocalParts::LocalParts(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventContoursOutput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_LocalPartsShowDebug, this)
{
  initRandomNumbers();
}

// ######################################################################
LocalParts::~LocalParts()
{
}

// ######################################################################
void LocalParts::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event --%s-- %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (strcmp(e->getWinName(), "LocalParts"))
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


void LocalParts::onSimEventContoursOutput(SimEventQueue& q, rutz::shared_ptr<SimEventContoursOutput>& e)
{

  itsContours = e->getContours();

  //for(uint i=0; i<itsContours.size(); i++)
  //{

  //  Image<PixRGB<byte> > tmp(320,240,ZEROS);


  //  for(uint j=0; j<itsContours[i].size(); j++)
  //  {
  //    tmp.setVal(itsContours[i].points[j], PixRGB<byte>(255,0,0));
  //    float ori = itsContours[i].ori[j];
  //    if (ori < 0) ori += M_PI;
  //    if (ori >= M_PI) ori -= M_PI;
  //  }

  //  std::vector<Point2D<int> > res = approxPolyDP(itsContours[i].points, 3);

  //  for(uint j=0; j<res.size()-2; j++)
  //  {
  //    Image<PixRGB<byte> > tmp2 = tmp;
  //    drawLine(tmp2, res[j+1], res[j], PixRGB<byte>(0,255,0));
  //    drawLine(tmp2, res[j+1], res[j+2], PixRGB<byte>(0,255,0));
  //    double ang = angle(res[j+1], res[j], res[j+2]);
  //    LINFO("Ang %f %f", acos(ang), acos(ang)*180/M_PI);
  //    //drawCircle(tmp, res[j], 3, PixRGB<byte>(0,255,0));
  //    SHOWIMG(tmp2);
  //  }

  //}

  /*
  Dims dims(320,240);

  itsLinesMag = Image<float>(dims, ZEROS);
  itsLinesOri = Image<float>(dims, ZEROS);
  for(uint i=0; i<itsContours.size(); i++)
  {
    for(uint j=0; j<itsContours[i].size(); j++)
    {
      itsLinesMag.setVal(itsContours[i].points[j], 255.0);

      float ori = itsContours[i].ori[j];
      if (ori < 0) ori += M_PI;
      if (ori >= M_PI) ori -= M_PI;
      itsLinesOri.setVal(itsContours[i].points[j], ori);
    }
  }
  inplaceNormalize(itsLinesMag, 0.0F, 100.0F);
  itsEdgesDT = chamfer34(itsLinesMag, 10.0F);

  int numOfEntries = 60;
  double D = M_PI/numOfEntries;
  //seperate each ori into its own map
  itsOriEdgesDT = ImageSet<float>(numOfEntries, itsLinesMag.getDims(), ZEROS);
  for(uint i=0; i<itsLinesMag.size(); i++)
  {
    if (itsLinesMag[i] > 0)
    {
      float ori = itsLinesOri[i];
      int oriIdx = (int)floor(ori/D);
      itsOriEdgesDT[oriIdx].setVal(i,itsLinesMag[i]);
    }
  }

  //Take the distance transform of each ori bin
  for(uint i=0; i<itsOriEdgesDT.size(); i++)
    itsOriEdgesDT[i] = chamfer34(itsOriEdgesDT[i], 30.0F);

  evolve(q);
  */
}


// ######################################################################
void LocalParts::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage(q);
      if (disp.initialized())
        ofs->writeRgbLayout(disp, "LocalParts", FrameInfo("LocalParts", SRC_POS));
    }
}


// ######################################################################
void LocalParts::evolve(SimEventQueue& q)
{

  /*
  //std::vector<SurfaceState> surfaces = proposeSurfaces();
  std::vector<SurfaceState> surfaces = itsSurfaces;

  for(uint i=0; i<surfaces.size(); i++)
  {
    calcSurfaceLikelihood(surfaces[i]);
  }
  itsProposals = surfaces;

  //itsSurfaces = surfaces;

  //sort by prob
  //Update the surfaces
  for(uint i=0; i<surfaces.size(); i++)
  {
    if (surfaces[i].prob > itsSurfaces[i].prob)
      itsSurfaces[i] = surfaces[i];
  }

  q.post(rutz::make_shared(new SimEventLocalPartsOutput(this, itsSurfaces)));


  std::vector<Contours::Contour> contourBias; 

  for(uint sid=0; sid<itsSurfaces.size(); sid++)
  {
    for(uint cid=0; cid<itsContours.size(); cid++)
    {
      Image<float> tmp(320,240,ZEROS);


      drawSuperquadric(tmp,
          Point2D<int>(itsSurfaces[sid].pos),
          itsSurfaces[sid].a, itsSurfaces[sid].b, itsSurfaces[sid].e, 
          2.0F,
          itsSurfaces[sid].rot, itsSurfaces[sid].k1, itsSurfaces[sid].k2,
          itsSurfaces[sid].start,itsSurfaces[sid].end,3);

      Contours::Contour& contour = itsContours[cid];
      for(uint i=0; i<contour.size(); i++)
        tmp.setVal(contour.points[i], tmp.getVal(contour.points[i]) - 1.0F);

      //Find the overlap and mark that contour with 
      int overCnt = 0;
      Contours::Contour biasContour;
      for(int y=0; y<tmp.getHeight(); y++)
        for(int x=0; x<tmp.getWidth(); x++)
        {
          if (tmp.getVal(x,y) == 1.0F)
            overCnt++;

          if (tmp.getVal(x,y) > 0)
            biasContour.points.push_back(Point2D<int>(x,y));
        }

      if (overCnt > 20)
        contourBias.push_back(biasContour);

    }

  }

  q.post(rutz::make_shared(new SimEventContoursBias(this, contourBias)));

  */

}

std::vector<LocalParts::PartState> LocalParts::proposeParts()
{
  std::vector<PartState> parts;

  return parts;
}

void LocalParts::calcPartLikelihood(PartState& part)
{
}

Layout<PixRGB<byte> > LocalParts::getDebugImage(SimEventQueue& q)
{
  Layout<PixRGB<byte> > outDisp;

  return outDisp;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

