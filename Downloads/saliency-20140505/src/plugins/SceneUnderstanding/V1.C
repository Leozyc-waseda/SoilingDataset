/*!@file SceneUnderstanding/V1.C  */


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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V1.C $
// $Id: V1.C 14350 2010-12-28 20:01:35Z lior $
//

#ifndef V1_C_DEFINED
#define V1_C_DEFINED

#include "plugins/SceneUnderstanding/V1.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
#include "Simulation/SimEventQueue.H"
#include "Neuro/EnvVisualCortex.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_V1 = {
  MOC_SORTPRI_3,   "V1-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_V1ShowDebug =
  { MODOPT_ARG(bool), "V1ShowDebug", &MOC_V1, OPTEXP_CORE,
    "Show debug img",
    "v1-debug", '\0', "<true|false>", "false" };


//Define the inst function name
SIMMODULEINSTFUNC(V1);

// ######################################################################
V1::V1(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventLGNOutput),
  SIMCALLBACK_INIT(SimEventV1Bias),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_V1ShowDebug, this),
  itsThreshold(0.10),
  itsBiasThreshold(0.05),
  itsAngBias(0)

{
  itsAttenLoc.i = -1;
  itsAttenLoc.j = -1;

  //itsAttenLoc.i = 892;
  //itsAttenLoc.j = 332;
  
  //itsAttenLoc.i = 467;
  //itsAttenLoc.j = 27;

  itsWinSize = Dims(320,240);



}

// ######################################################################
V1::~V1()
{

}

// ######################################################################
void V1::onSimEventLGNOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventLGNOutput>& e)
{
  itsTimer.reset();
  itsLGNData = e->getCells();

  //Dims imgSize = itsLGNData[0].getDims();
  //for(int y=0; y<imgSize.h(); y+=25)
  //  for(int x=0; x<imgSize.w(); x+=25)
  //  {
  //    itsSpatialBias.push_back(SpatialBias(x,y,50,50, 0.10));
  //  }

  evolve(q);

  

}

// ######################################################################
void V1::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "V1", FrameInfo("V1", SRC_POS));
    }
}


void V1::setBias(const Image<float> &biasImg)
{

}

// ######################################################################
void V1::onSimEventV1Bias(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV1Bias>& e)
{
  itsSpatialBias = e->getSpatialBias();
  itsTimer.mark();
  printf("Total time %0.2f sec\n", itsTimer.real_secs());
  fflush(stdout);

  evolve(q);

  //LINFO("Show V1");
  //Layout<PixRGB<byte> > layout = getDebugImage();
  //Image<PixRGB<byte> > tmp = layout.render();
  //SHOWIMG(tmp);
  
}

void V1::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{
  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());


  if (strcmp(e->getWinName(), "V1"))
    return;

  switch(e->getKey())
  {
    case 10: //1
      itsThreshold += 0.01;
      if (itsThreshold > 1) itsThreshold = 1;
      break;
    case 24: //q
      itsThreshold -= 0.01;
      if (itsThreshold < 0) itsThreshold = 0;
      break;
    case 11: //1
      itsBiasThreshold += 0.01;
      if (itsBiasThreshold > 1) itsBiasThreshold = 1;
      break;
    case 25: //q
      itsBiasThreshold -= 0.01;
      if (itsBiasThreshold < 0) itsBiasThreshold = 0;
      break;
    default:
      break;
  }

  
  if (e->getMouseClick().isValid())
  {
    LINFO("Set spatial bias");
    itsAttenLoc = e->getMouseClick();
    //itsSpatialBias.loc = e->getMouseClick();
    //itsSpatialBias.dims = Dims(50,50);
    //itsSpatialBias.threshold = itsBiasThreshold;
  }

  evolve(q);

}


// ######################################################################
void V1::evolve(SimEventQueue& q)
{
  //evolveGabor();
 // evolveSobel();
//  evolveCanny();
    evolveTensor();
    //Layout<PixRGB<byte> > layout = getDebugImage();
    //Image<PixRGB<byte> > tmp = layout.render();
    //SHOWIMG(tmp);

    q.post(rutz::make_shared(new SimEventV1Output(this, itsEdgesState)));
}

void V1::evolveTensor()
{
  Image<float> d1;
  Image<float> d2;
  Image<float> d3;

  if (itsAttenLoc.isValid())
  {
    d1 = crop(itsLGNData[0], itsAttenLoc, itsWinSize);
    d2 = crop(itsLGNData[1], itsAttenLoc, itsWinSize);
    d3 = crop(itsLGNData[2], itsAttenLoc, itsWinSize);
  } else {
    d1 = itsLGNData[0];
    d2 = itsLGNData[1];
    d3 = itsLGNData[2];
  }


  itsInput = d1;


  itsEdgesState.lumTensorField = getTensor(d1,3);
  itsEdgesState.rgTensorField  = getTensor(d2,3);
  itsEdgesState.byTensorField  = getTensor(d3,3);

  //Non maximal supperssion
  nonMaxSurp(itsEdgesState.lumTensorField); 
  nonMaxSurp(itsEdgesState.rgTensorField);
  nonMaxSurp(itsEdgesState.byTensorField);


  LINFO("Bias size %i", (int)itsSpatialBias.size());
  ////Extract edges by keeping only the edges with values greater
  ////then 10% of the max mag.
  applyThreshold(itsEdgesState.lumTensorField, itsSpatialBias);
  applyThreshold(itsEdgesState.rgTensorField, itsSpatialBias);
  applyThreshold(itsEdgesState.byTensorField, itsSpatialBias);

}

void V1::applyThreshold(TensorField& tensorField, std::vector<SpatialBias>& spatialBias)
{
  
  Image<float> mag = getTensorMag(tensorField);
  float min, max;
  getMinMax(mag, min,max);

  for(int y=0; y<mag.getHeight(); y++)
    for(int x=0; x<mag.getWidth(); x++)
    {
      bool biased = false;

      for(uint i=0;  i<spatialBias.size(); i++)
      {
        if (spatialBias[i].contains(x,y))
        {
          if (mag.getVal(x,y) < max*spatialBias[i].threshold)
            tensorField.setVal(x,y,0);
          biased = true;
        }
      }

      if (!biased)
      {
        if (mag.getVal(x,y) < max*itsThreshold)
          tensorField.setVal(x,y,0);
      }

    }
}

//// ######################################################################
//void V1::evolveSobel()
//{
//
//  Image<float> magImg, oriImg;
//  //for(uint i=0; i<itsV1CellsInput.size(); i++)
//
//  for(uint i=0; i<1; i++)
//  {
//    gradientSobel(itsV1CellsInput[i], magImg, oriImg);
//    Image<float> edgeImg(magImg.getDims(), ZEROS);
//
//    itsEdgesState.clear();
//
//    for(int y=0; y<magImg.getHeight(); y++)
//      for(int x=0; x<magImg.getWidth(); x++)
//      {
//        float edgeProb = magImg.getVal(x,y)/200; //1.0F/(1.0F + expf(0.09*(30.0-magImg.getVal(x,y))));
//        if (edgeProb > 1.0) edgeProb = 1.0;
//        if (edgeProb > 0.0)
//        {
//          EdgeState edgeState;
//          edgeState.pos = Point2D<int>(x,y);
//          edgeState.ori = oriImg.getVal(x,y);
//          edgeState.var = (10*M_PI/180)*(10*M_PI/180); //10
//          edgeState.prob = edgeProb;
//
//          itsEdgesState.push_back(edgeState);
//        }
//
//        //Build the edgeDistance with a threshold
//        if (edgeProb > 0.25)
//          edgeImg.setVal(Point2D<int>(x,y), 1.0F);
//
//      }
//    //itsEdgesDT = chamfer34(edgeImg, 50.0F); //get the distance to edges max at 50pixels
//    itsEdgesDT = saliencyChamfer34(edgeImg); //get the distance to edges max at 50pixels
//    itsEdgesOri = oriImg;
//  }
//}
//
//void V1::evolveCanny()
//{
//  Image<float> magImg, oriImg;
//  //for(uint i=0; i<itsV1CellsInput.size(); i++)
//
//  for(uint i=0; i<1; i++)
//  {
//    gradientSobel(itsV1CellsInput[i], magImg, oriImg);
//    Image<float> edgeImg(magImg.getDims(), ZEROS);
//
//    inplaceNormalize(itsV1CellsInput[i], 0.0F, 255.0F);
//    Image<byte> in = itsV1CellsInput[i];
//    Image<byte> edges(in.getDims(), ZEROS);
//    cvCanny(img2ipl(in), img2ipl(edges), 50, 100);
//
//    itsEdgesState.clear();
//
//    for(int y=0; y<edges.getHeight(); y++)
//      for(int x=0; x<edges.getWidth(); x++)
//      {
//        if (edges.getVal(x,y) > 0)
//        {
//          float edgeProb = magImg.getVal(x,y)/200; //1.0F/(1.0F + expf(0.09*(30.0-magImg.getVal(x,y))));
//          if (edgeProb > 1.0) edgeProb = 1.0;
//          if (edgeProb > 0.0)
//          {
//            EdgeState edgeState;
//            edgeState.pos = Point2D<int>(x,y);
//            edgeState.ori = oriImg.getVal(x,y);
//            edgeState.var = (10*M_PI/180)*(10*M_PI/180); //10
//            edgeState.prob = edgeProb;
//
//            itsEdgesState.push_back(edgeState);
//          }
//
//          //Build the edgeDistance with a threshold
//         // if (edgeProb > 0.25)
//          edgeImg.setVal(Point2D<int>(x,y), 1.0F);
//        }
//
//      }
//    //itsEdgesDT = chamfer34(edgeImg, 50.0F); //get the distance to edges max at 50pixels
//    itsEdgesDT = saliencyChamfer34(edgeImg); //get the distance to edges max at 50pixels
//    itsEdgesOri = oriImg;
//  }
//}
//
//
//// ######################################################################
//void V1::evolveGabor()
//{
//
// // float filter_period = 100;
// // float elongation = 2.0;
// // float angle = 90;
// // int size = -1;
// // const double major_stddev = filter_period / 30.0;
// // const double minor_stddev = major_stddev * elongation;
//
//  //Image<float> gabor0 = gaborFilter3(major_stddev, minor_stddev,
//  //    filter_period, 90, 180 - angle + 90.0 , size);
//  //Image<float> gabor90 = gaborFilter3(major_stddev, minor_stddev,
//  //    filter_period, 0, 180 - angle + 90.0 , size);
//
// Image<float> gabor0 = gaborFilter<float>(5.0F, //stdev
//                                   1.08, //period
//                                   0.0F, //phase
//                                   90.0F); //theta
//// Image<float> gabor90 = gaborFilter<float>(5.0F, //stdev
////                                   1.08, //period
////                                   0.0F, //phase
////                                   0.0F); //theta
////
//
//  // normalize to unit sum-of-squares:
//  gabor0 -= mean(gabor0); gabor0 /= sum(squared(gabor0));
////  gabor90 -= mean(gabor90); gabor90 /= sum(squared(gabor90));
//
//  Image<float> f0 = optConvolve(itsV1CellsInput[0], gabor0);
// // SHOWIMG(f0);
//  //Image<float> f90 = optConvolve(itsV1CellsInput[0], gabor90);
//  //Image<float> out = sqrt(squared(f0) + squared(f90));
//
//  //SHOWIMG(f0);
//  //SHOWIMG(f90);
//
//  //SHOWIMG(out);
//  Point2D<int> maxPos; float maxVal;
//  findMax(f0, maxPos, maxVal);
//  LINFO("Max at %i,%i %f", maxPos.i, maxPos.j, maxVal);
//  printf("%f;\n", maxVal);
//  fflush(stdout);
//
//  //SHOWIMG(out);
//
//}


Layout<PixRGB<byte> > V1::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  EigenSpace eigen = getTensorEigen(itsEdgesState.lumTensorField);
  Image<float> lumFeatures = eigen.l1-eigen.l2;

  eigen = getTensorEigen(itsEdgesState.rgTensorField);
  Image<float> rgFeatures = eigen.l1-eigen.l2;

  eigen = getTensorEigen(itsEdgesState.byTensorField);
  Image<float> byFeatures = eigen.l1-eigen.l2;

  inplaceNormalize(lumFeatures, 0.0F, 255.0F);
  inplaceNormalize(rgFeatures, 0.0F, 255.0F);
  inplaceNormalize(byFeatures, 0.0F, 255.0F);

  //SHOWIMG(lumFeatures);

  Image<PixRGB<byte> > attnInput = itsInput; //itsLGNData[0];

  Image<PixRGB<byte> > input = itsLGNData[0];
  if (itsAttenLoc.isValid())
  {
    drawRect(input, Rectangle(itsAttenLoc, itsWinSize), PixRGB<byte>(0,255,0), 3);
    input = rescale(input, attnInput.getDims());
  }

  char msg[255];
  sprintf(msg, "T: %0.2f BT: %0.2f", itsThreshold*100, itsBiasThreshold*100);
  writeText(attnInput, Point2D<int>(0,0), msg,
      PixRGB<byte>(255,255,255),
      PixRGB<byte>(0,0,0));

  for(uint i=0; i<itsSpatialBias.size(); i++)
  {
    Rectangle rect = Rectangle::centerDims(itsSpatialBias[i].loc, itsSpatialBias[i].dims);
    if (attnInput.rectangleOk(rect))
      drawRect(attnInput,rect ,
          PixRGB<byte>(255,0,0));
  }


  
  outDisp = hcat(input, attnInput);
  outDisp = hcat(outDisp, toRGB(Image<byte>(lumFeatures)));
  outDisp = hcat(outDisp, toRGB(Image<byte>(rgFeatures)));
  outDisp = hcat(outDisp, toRGB(Image<byte>(byFeatures)));

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

