/*!@file plugins/SceneUnderstanding/LineFitting.C  */

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

#ifndef LineFitting_C_DEFINED
#define LineFitting_C_DEFINED

#include "Image/OpenCVUtil.H" //Need to be first to avoid type def conf
#include "plugins/SceneUnderstanding/LineFitting.H"
#include "plugins/SceneUnderstanding/V2.H"

#include "Image/DrawOps.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueue.H"
#include "GUI/DebugWin.H"
#include "plugins/SceneUnderstanding/LFLineFitter/LFLineFitter.h"


const ModelOptionCateg MOC_LineFitting = {
  MOC_SORTPRI_3,   "LineFitting-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_LineFittingShowDebug =
  { MODOPT_ARG(bool), "LineFittingShowDebug", &MOC_LineFitting, OPTEXP_CORE,
    "Show debug img",
    "linefitting-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(LineFitting);


// ######################################################################
LineFitting::LineFitting(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV1Output),
  SIMCALLBACK_INIT(SimEventInputFrame),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_LineFittingShowDebug, this)
{
}

// ######################################################################
LineFitting::~LineFitting()
{
}

// ######################################################################
void LineFitting::onSimEventV1Output(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV1Output>& e)
{
  //itsV1EdgesState = e->getEdgesState();

  //evolve(q);
}


// ######################################################################
void LineFitting::onSimEventInputFrame(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventInputFrame>& e)
{
  // here is the inputs image:
  GenericFrame frame = e->frame();

  itsInImage = frame.asRgb();

  evolve(q);
}

void LineFitting::evolve(SimEventQueue& q)
{

  Image<float> edges = luminance(itsInImage);
  itsLines = FitLine(edges);

  q.post(rutz::make_shared(new SimEventV2Output(this, itsLines, itsInImage.getDims())));

}


std::vector<V2::LineSegment> LineFitting::FitLine(const Image<float>& edges)
{
  std::map<int,Point2D<int> > edgesMap;

  LFLineFitter lf;
  lf.Configure("para_line_fitter.txt");
	lf.Init();
	lf.FitLine(img2ipl(edges));

  int nLines = lf.rNLineSegments();
  LFLineSegment* lineSegmentMap = lf.rOutputEdgeMap();

  LFLineSegment* lines = new LFLineSegment[nLines];

  std::vector<V2::LineSegment> lineSegments;
  for (int i=0 ; i<nLines ; i++)
  {
    lines[i] = lineSegmentMap[i];
    lineSegments.push_back(V2::LineSegment(
        Point2D<float>(lines[i].sx_, lines[i].sy_),
        Point2D<float>(lines[i].ex_, lines[i].ey_)));
  }

  return lineSegments;


}


// ######################################################################
void LineFitting::onSimEventSaveOutput(SimEventQueue& q,
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
        ofs->writeRgbLayout(disp, "LineFitting", FrameInfo("LineFitting", SRC_POS));
    }
}


Layout<PixRGB<byte> > LineFitting::getDebugImage()
{

  Layout<PixRGB<byte> > disp;

  Image<PixRGB<byte> > tmp = itsInImage;
  for(uint i=0; i<itsLines.size(); i++)
    drawLine(tmp, (Point2D<int>)itsLines[i].p1, (Point2D<int>)itsLines[i].p2, PixRGB<byte>(0,255,0));

  disp = hcat(itsInImage, tmp);

  usleep(10000);
  return disp;


}

#endif
