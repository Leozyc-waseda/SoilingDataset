/*!@file SceneUnderstanding/Regions.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Regions.C $
// $Id: Regions.C 13765 2010-08-06 18:56:17Z lior $
//

#ifndef Regions_C_DEFINED
#define Regions_C_DEFINED

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
#include "Simulation/SimEventQueue.H"
#include "plugins/SceneUnderstanding/Regions.H"
#include "Neuro/EnvVisualCortex.H"
#include "GUI/DebugWin.H"
#include "Util/MathFunctions.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_Regions = {
  MOC_SORTPRI_3,   "Regions-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_RegionsShowDebug =
  { MODOPT_ARG(bool), "RegionsShowDebug", &MOC_Regions, OPTEXP_CORE,
    "Show debug img",
    "regions-debug", '\0', "<true|false>", "false" };


// ######################################################################
Regions::Regions(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventLGNOutput),
  SIMCALLBACK_INIT(SimEventSMapOutput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  SIMCALLBACK_INIT(SimEventRegionsPrior),
  itsShowDebug(&OPT_RegionsShowDebug, this)

{

  itsTempColor = 104;


  itsRegionsState.clear();
}

// ######################################################################
Regions::~Regions()
{

}

// ######################################################################
void Regions::onSimEventLGNOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventLGNOutput>& e)
{
  itsRegionsCellsInput = e->getCells();
  inplaceNormalize(itsRegionsCellsInput[0], 0.0F, 255.0F);
  evolve();

  q.post(rutz::make_shared(new SimEventRegionsOutput(this, itsRegionsState)));

}

// ######################################################################
void Regions::onSimEventRegionsPrior(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventRegionsPrior>& e)
{
  itsRegionsPrior = e->getRegions();
  evolve();
}


// ######################################################################
void Regions::onSimEventSMapOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventSMapOutput>& e)
{
  itsSMapInput = e->getCells();

  //for(uint i=0; i<1; i++)
  //Add the background
  Image<float> tmp(itsRegionsCellsInput[0].getDims(), ZEROS);
  if (itsSMapInput.size() > 0)
  {
    SMap::SMapState sms = itsSMapInput[0];

    itsBackgroundRegion.mu = sms.mu;
    itsBackgroundRegion.sigma = sms.sigma;

    for(uint pix=0; pix<sms.region.size(); pix++)
    {
      //only take pixels that are close to the mean
      //if ( gauss<double>((double)itsRegionsCellsInput[0].getVal(sms.region[pix]),
      //      itsBackgroundRegion.mu, itsBackgroundRegion.sigma) > 1.0e-20)
      {
        itsBackgroundRegion.region.push_back(sms.region[pix]);
        tmp.setVal(sms.region[pix], 255);
      }
    }

    calcRegionLikelihood(itsBackgroundRegion);
  }

  ////Find the edge of the background
  //Image<float> mag, ori;
  //gradientSobel(tmp, mag, ori);

  //std::vector<Point2D<int> > boundry;
  //for(int y=0; y<tmp.getHeight(); y++)
  //  for(int x=0; x<tmp.getWidth(); x++)
  //    if (mag.getVal(x,y) > 0)
  //      boundry.push_back(Point2D<int>(x,y));


  //if (itsSMapInput.size() > 1)
  //{
  //  for(uint i=1; i<itsSMapInput.size(); i++)
  //  {
  //    SMap::SMapState sms = itsSMapInput[i];

  //    RegionState rs;
  //    rs.mu = sms.mu;
  //    rs.sigma = sms.sigma;

  //    for(uint pix=0; pix<sms.region.size(); pix++)
  //    {
  //      double backgroundProb = gauss<double>((double)itsRegionsCellsInput[0].getVal(sms.region[pix]), itsRegionsState[0].mu, itsRegionsState[0].sigma);
  //      //double foregroundProb = gauss<double>((double)itsRegionsCellsInput[0].getVal(sms.region[pix]), rs.mu, rs.sigma);

  //      //check if this pixel will be better off beloging to the background
  //      if ((backgroundProb > 1.0e-2) )
  //      {
  //        //check the distance from this pixel to the rest;
  //        double minDist = 100;
  //        for(uint j=0; j<boundry.size(); j++)
  //        {
  //          double dist = sms.region[pix].distance(boundry[j]);
  //          if (dist < minDist)
  //            minDist = dist;
  //        }

  //        if (minDist < 3)
  //        {
  //          itsRegionsState[0].region.push_back(sms.region[pix]);
  //          boundry.push_back(sms.region[pix]);
  //        }
  //        else
  //          rs.region.push_back(sms.region[pix]);

  //      }
  //      else
  //        rs.region.push_back(sms.region[pix]);

  //    }

  //    calcRegionLikelihood(rs);

  //    itsRegionsState.push_back(rs);
  //  }
  //}



}

// ######################################################################
void Regions::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "Regions", FrameInfo("Regions", SRC_POS));
    }
}


// ######################################################################
void Regions::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  //if (e->getWinName() == "Regions")
  {
    LINFO("Got event %s %ix%i key=%i",
        e->getWinName(),
        e->getMouseClick().i,
        e->getMouseClick().j,
        e->getKey());

    switch(e->getKey())
    {
      case 54: //c for clear
        itsUserPolygon.clear();
        break;
      case 36: //enter for closing polygon
        break;
      case 98: //up key
        itsTempColor += 1;
        break;
      case 104: //down key
        itsTempColor -= 1;
        break;
    }
    if (e->getMouseClick().isValid())
    {
      itsUserPolygon.push_back(e->getMouseClick());
      LINFO("Color is %f", itsRegionsCellsInput[0].getVal(e->getMouseClick()));
    }
    evolve();

  }

}



void Regions::calcRegionLikelihood(RegionState& rs)
{

  double minProb = 1.0e-5;

  rs.prob = log(minProb)*512*512;  //set the initial prob to very low

  if (itsRegionsCellsInput.size() > 0)
  {
    for(uint i=0; i<rs.region.size(); i++)
      rs.prob += log(gauss<double>((double)itsRegionsCellsInput[0].getVal(rs.region[i]), rs.mu, rs.sigma));

    //Correct for more data
    rs.prob -= log(minProb) * rs.region.size();
  }

}


double Regions::calcRegionLikelihood(Image<float>& mu)
{

  double prob = 0;
  Image<float> in = itsRegionsCellsInput[0];

  for(uint i=0; i<mu.size(); i++)
  {
    prob += log(gauss<double>((double)in[i], (double)mu[i], 5));
  }

  return prob;
}


// ######################################################################
void Regions::evolve()
{


  //for(uint py=0; py<240; py++)
  //  for(uint px=0;px<320; px++)
  //  {
  //    RegionState newRs = itsRegionsState[0];
  //    //Pick a point at random and add it to the region
  //   // int px = randomUpToIncluding(320-1);
  //   // int py = randomUpToIncluding(240-1);
  //    LINFO("New pix %ix%i", px, py);
  //    newRs.region.push_back(Point2D<int>(px,py));


  //    calcRegionLikelihood(newRs);
  //    LINFO("New Likelihood %e old likelihood %e", newRs.prob, itsRegionsState[0].prob);

  //    if (newRs.prob > itsRegionsState[0].prob)
  //    {
  //      itsRegionsState[0] = newRs;
  //      LINFO("Taking new point");
  //    }
  //  }

  //Calculate region likelihood
  itsRegionsState = itsRegionsPrior;
  //for(uint i=0; i<itsRegionsPrior.size(); i++)
  //{
  //  RegionState newRs = itsRegionsPrior[i];
  //  calcRegionLikelihood(newRs);
  //  itsRegionsState[i] = newRs;

  //  LINFO("%i, New Likelihood %e old likelihood %e",
  //      i, newRs.prob, itsRegionsState[i].prob);
  //  //if (newRs.prob > itsRegionsState[i].prob)
  //  //{
  //  //}
  //}

  //RegionsState rs;
  //rs.mu = itsTempColor;
  //rs.sigma = 5;
  //for(uint j=0; j<itsRegionsCellsInput[0].getHeight(); j++)
  //  for(uint i=0; i<itsRegionsCellsInput[0].getWidth(); i++)
  //  {
  //    RegionsState tmpRs = rs;

  //    tmpRs.region.push_back(Point2D<int>(i,j));
  //    calcRegionLikelihood(rs);

  //  }


}

Layout<PixRGB<byte> > Regions::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  for(int i=0; i<1; i++)
  {
    Image<float> in = itsRegionsCellsInput[i];
    //inplaceNormalize(in, 0.0F, 255.0F);

 //   Image<float> perc(in.getDims(), ZEROS);
     Image<PixRGB<byte> > perc = in;

     //Show the background


     for(uint pix=0; pix<itsBackgroundRegion.region.size(); pix++)
     {
       perc.setVal(itsBackgroundRegion.region[pix], PixRGB<byte>(0,0,0)); //itsBackgroundRegion.mu);
     }

    for(uint i=0; i<itsRegionsState.size(); i++)
    {
      RegionState rs = itsRegionsState[i];
//      LINFO("Region %i mu=%0.2f sigma=%0.2f prob=%e ",
//          i, rs.mu, rs.sigma, rs.prob);
      for(uint pix=0; pix<rs.region.size(); pix++)
      {
        perc.setVal(rs.region[pix], PixRGB<byte>(255,0,0));
      }
    }

    //in += perc;

    //Draw ponts for user polygon
    for(uint i=0; i<itsUserPolygon.size(); i++)
    {
      drawCircle(in, itsUserPolygon[i], 3, 255.0F);
    }


    outDisp = hcat(toRGB(Image<byte>(in)), perc);
  }

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

