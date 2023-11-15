/*!@file SceneUnderstanding/CornersFeatures.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/CornersFeatures.C $
// $Id: CornersFeatures.C 13765 2010-08-06 18:56:17Z lior $
//

#ifndef CornersFeatures_C_DEFINED
#define CornersFeatures_C_DEFINED

#include "plugins/SceneUnderstanding/CornersFeatures.H"

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
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_CornersFeatures = {
  MOC_SORTPRI_3,   "CornersFeatures-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_CornersFeaturesShowDebug =
  { MODOPT_ARG(bool), "CornersFeaturesShowDebug", &MOC_CornersFeatures, OPTEXP_CORE,
    "Show debug img",
    "corners-debug", '\0', "<true|false>", "false" };


//Define the inst function name
SIMMODULEINSTFUNC(CornersFeatures);

// ######################################################################
CornersFeatures::CornersFeatures(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV2Output),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_CornersFeaturesShowDebug, this),
  itsPatchSize(20,20)
{

  initRandomNumbers();

  itsSOFM = new SOFM("Corners", itsPatchSize.sz(), 50,50 );
  //itsSOFM->RandomWeights();
  itsSOFM->ReadNet("SOFM.net");

}


// ######################################################################
CornersFeatures::~CornersFeatures()
{
}


// ######################################################################
void CornersFeatures::onSimEventV2Output(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV2Output>& e)
{
  itsLines = e->getLines();
  //itsCornersProposals = e->getCornersProposals();
  evolve(q);

  q.post(rutz::make_shared(new SimEventCornersOutput(this, itsCorners)));
}

// ######################################################################
void CornersFeatures::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage(q);
      ofs->writeRgbLayout(disp, "CornersFeatures", FrameInfo("CornersFeatures", SRC_POS));
    }
}

// ######################################################################
void CornersFeatures::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (e->getMouseClick().isValid())
  {
  }

  //evolve(q);

}

std::vector<CornersFeatures::CornerState> CornersFeatures::getCorners(std::vector<V2::LineSegment>& lines)
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
    V2::LineSegment& ls1 = itsLines[i];

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
      V2::LineSegment& ls2 = itsLines[j];

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

    CornerState c1;
    c1.center = ls1.p1;
    c1.angles = angles1;
    corners.push_back(c1);

    CornerState c2;
    c2.center = ls1.p2;
    c2.angles = angles2;
    corners.push_back(c2);



  }


  return corners;

}

// ######################################################################
void CornersFeatures::evolve(SimEventQueue& q)
{

  //Find corners from lines end points;


  std::vector<CornerState> corners = getCorners(itsLines);

  itsCorners = corners;

  //Add to corners db
  ////////////////////////////////////////

  //for(uint i=0; i<corners.size(); i++)
  //{
  //  CornerState& cs = corners[i];
  //  //Find the nearest corner in the db
  //  // and update the prob
  //  //
  //  std::vector<GaussianDef> gmmF;
  //  double weight = 1.0/double(cs.angles.size()); //equal weight
  //  for(uint i=0; i<cs.angles.size(); i++)
  //    gmmF.push_back(GaussianDef(weight, cs.angles[i], 1*M_PI/180)); //1 deg variance

  //  bool found = false;
  //  for(uint j=0; j<itsCornersDB.size(); j++)
  //  {
  //    CornerState& dbcs = itsCornersDB[j];

  //    std::vector<GaussianDef> gmmG;
  //    double weight = 1.0/double(dbcs.angles.size()); //equal weight
  //    for(uint i=0; i<dbcs.angles.size(); i++)
  //      gmmG.push_back(GaussianDef(weight, dbcs.angles[i], 1*M_PI/180)); //1 deg variance

  //    double dist = L2GMM(gmmF, gmmG);

  //    if (dist < 2)
  //    {
  //      //Found it
  //      dbcs.prob++;
  //      found = true;
  //      break;
  //    }

  //  }

  //  if (!found)
  //    itsCornersDB.push_back(cs); //add it to the db
  //}


  ////Write out the corners db
  //FILE *fp = fopen("corners.dat", "w");

  //for(uint i=0; i<itsCornersDB.size(); i++)
  //{
  //  CornerState& dbcs = itsCornersDB[i];
  //  for(uint j=0; j<dbcs.angles.size(); j++)
  //    fprintf(fp," %f", dbcs.angles[j]);
  //  fprintf(fp," %f\n", dbcs.prob);
  //}
  //fclose(fp);




  ////////////////////////////////////////

  //Sample from the cor
  //Image<float> tmp=itsCornersProposals;
  //inplaceNormalize(itsCornersProposals, 0.0F, 100.0F);

  //inplaceNormalize(itsEdges, 0.0F, 255.0F);

  //itsCorners.clear();

  //int radius = 10;
  //for(int i=0; i<20; i++)
  //{
  //  Point2D<int> maxLoc; float maxVal;
  //  findMax(tmp, maxLoc, maxVal);

  //  Point2D<int> tl(maxLoc.i-(itsPatchSize.w()/2), maxLoc.j-(itsPatchSize.h()/2));

  //  if (itsEdges.coordsOk(tl) &&
  //      itsEdges.coordsOk(tl.i+itsPatchSize.w(), tl.j+itsPatchSize.h()))
  //  {
  //    Image<float> patch = crop(itsEdges, tl, itsPatchSize);

  //    itsSOFM->setInput(patch);
  //    itsSOFM->Propagate();

  //    double winnerVal;
  //    Point2D<int> winnerId = itsSOFM->getWinner(winnerVal);
  //    //LINFO("WinnerVal %e\n", winnerVal);


  //    Image<float> learnedPatch(patch.getDims(), NO_INIT);
  //    std::vector<float> weights = itsSOFM->getWeights(winnerId);
  //    for(uint i=0; i<weights.size(); i++)
  //      learnedPatch[i] = weights[i];

  //    //Only organize if we have not learned this patch
  //    //if (winnerVal > 1)
  //    //  itsSOFM->organize(patch);

  //    CornerState cs;
  //    cs.id = winnerId;
  //    cs.patch = patch;
  //    cs.learnedPatch = learnedPatch;
  //    cs.loc = maxLoc;
  //    cs.prob = 0;
  //    itsCorners.push_back(cs);
  //  }
  //  //Apply IOR
  //  drawDisk(tmp, maxLoc, radius, 0.0f);
  //}

  //itsSOFMMap = itsSOFM->getWeightsImage();

  //itsSOFM->WriteNet("SOFM.net");

}


Layout<PixRGB<byte> > CornersFeatures::getDebugImage(SimEventQueue& q)
{
  Layout<PixRGB<byte> > outDisp;

  Image<PixRGB<byte> > cornersImg(320, 240, ZEROS);
  for(uint i=0; i<itsCorners.size(); i++)
  {
    for(uint ai=0; ai<itsCorners[i].angles.size(); ai++)
    {
      int x1 = int(cos(itsCorners[i].angles[ai])*30.0/2.0);
      int y1 = int(sin(itsCorners[i].angles[ai])*30.0/2.0);
      Point2D<float> p1(itsCorners[i].center.i-x1, itsCorners[i].center.j+y1);

      drawLine(cornersImg, Point2D<int>(itsCorners[i].center), Point2D<int>(p1), PixRGB<byte>(0,255,0));
    }
  }

  outDisp = cornersImg; 

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

