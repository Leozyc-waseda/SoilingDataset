/*!@file SceneUnderstanding/SMap.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/SMap.C $
// $Id: SMap.C 13551 2010-06-10 21:56:32Z itti $
//

#ifndef SMap_C_DEFINED
#define SMap_C_DEFINED

#include "plugins/SceneUnderstanding/SMap.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
#include "Image/ColorOps.H"
#include "Simulation/SimEventQueue.H"
#include "Neuro/EnvVisualCortex.H"
#include "GUI/DebugWin.H"
#include "Util/MathFunctions.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_SMap = {
  MOC_SORTPRI_3,   "SMap-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_SMapShowDebug =
  { MODOPT_ARG(bool), "SMapShowDebug", &MOC_SMap, OPTEXP_CORE,
    "Show debug img",
    "SMap-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(SMap);


// ######################################################################
SMap::SMap(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventInputFrame),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_SMapShowDebug, this)

{

  itsEvc = nub::soft_ref<EnvVisualCortex>(new EnvVisualCortex(mgr));
  addSubComponent(itsEvc);


}

// ######################################################################
SMap::~SMap()
{

}

// ######################################################################
void SMap::onSimEventInputFrame(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventInputFrame>& e)
{
  itsSMapCellsInput = e->frame().asRgb();
  evolve();

  q.post(rutz::make_shared(new SimEventSMapOutput(this, itsSMapState, itsSMap)));

}

// ######################################################################
void SMap::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "SMap", FrameInfo("SMap", SRC_POS));
    }
}


// ######################################################################
void SMap::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

}


// ######################################################################
void SMap::evolve()
{
  //SHOWIMG(itsSMapCellsInput);

  //CvMemStorage *storage2 = cvCreateMemStorage(1000); //TODO need to clear memory
  //CvSeq *comp;
  //int level = 1;
  //int threshold1 = 50;
  //int threshold2 = 20;
  //Image<PixRGB<byte> > dst(itsSMapCellsInput.getDims(), ZEROS);

  //cvPyrSegmentation(img2ipl(itsSMapCellsInput), img2ipl(dst),
  //    storage2, &comp, level, threshold1+1, threshold2+1);

  ////SHOWIMG(dst);



  //Image<byte> input = luminance(itsSMapCellsInput);
  ////SHOWIMG(input);
  //IplImage* img = img2ipl(input);
  //IplImage* marker_mask = cvCreateImage(cvGetSize(img), 8, 1);
  //cvCanny(img, marker_mask, 50, 100);

  //Image<byte> edges = ipl2gray(marker_mask);
  ////SHOWIMG(edges);
  //CvMemStorage* storage = cvCreateMemStorage(0);

  ////CvSeq* lines = cvHoughLines2(img2ipl(edges), storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 50, 0, 0);
  //CvSeq* lines = cvHoughLines2(img2ipl(edges), storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 250, 50, 10);


  //tmpEdges = edges; //Image<byte>(edges.getDims(), ZEROS);
  //Image<PixRGB<byte> > linesImg = edges;
  //for(int i=0; i<lines->total; i++)
  //{
  //    //float* line = (float*)cvGetSeqElem(lines,i);
  //    //float rho = line[0];
  //    //float theta = line[1];
  //    //CvPoint pt1, pt2;
  //    //double a = cos(theta), b = sin(theta);
  //    //double x0 = a*rho, y0 = b*rho;
  //    //pt1.x = cvRound(x0 + 1000*(-b));
  //    //pt1.y = cvRound(y0 + 1000*(a));
  //    //pt2.x = cvRound(x0 - 1000*(-b));
  //    //pt2.y = cvRound(y0 - 1000*(a));

  //    CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
  //    CvPoint pt1 = line[0];
  //    CvPoint pt2 = line[1];
  //
  //    drawLine(linesImg, Point2D<int>(pt1.x,pt1.y), Point2D<int>(pt2.x, pt2.y), PixRGB<byte>(255,0,0));
  //}

  ////SHOWIMG(linesImg);
  //

 //// LINFO("Calc vanishing points");
 //// vanishingPoints(lines, cvSize(img->width, img->height), 2*(img->width), 2*(img->height),
 ////     50, 000);



  //CvSeq* contours = 0;
  //cvFindContours( marker_mask, storage, &contours, sizeof(CvContour),
  //    CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

  //Image<PixRGB<byte> > contoursImg(edges.getDims(), ZEROS);
  //while( contours )
  //{
  //  // approximate contour with accuracy proportional
  //  // to the contour perimeter
  //  CvSeq* result =
  //    cvApproxPoly( contours, sizeof(CvContour), storage,
  //        CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
  //  // square contours should have 4 vertices after approximation
  //  // relatively large area (to filter out noisy contours)
  //  // and be convex.
  //  // Note: absolute value of an area is used because
  //  // area may be positive or negative - in accordance with the
  //  // contour orientation
  //  const double area = fabs(cvContourArea(result,CV_WHOLE_SEQ));
  //  if (result->total &&
  //      area >= 10) // && area <= 7000 &&
  //      //cvCheckContourConvexity(result))
  //  {
  //    //double s = 0;

  //    //for (int i = 0; i < 4; ++i)
  //    //{
  //    //  // find minimum angle between joint
  //    //  // edges (maximum of cosine)
  //    //  const double t =
  //    //    fabs(angle((CvPoint*)cvGetSeqElem( result, i % 4 ),
  //    //          (CvPoint*)cvGetSeqElem( result, (i-2) % 4 ),
  //    //          (CvPoint*)cvGetSeqElem( result, (i-1) % 4 )));
  //    //  s = s > t ? s : t;
  //    //}


  //    // if cosines of all angles are small
  //    // (all angles are ~90 degree) then write quandrangle
  //    // vertices to resultant sequence
  //    //if (s < mincos)
  //    {
  //      CvPoint *p1 = (CvPoint*)cvGetSeqElem( result, result->total-1 );
  //      for (int i = 0; i < result->total; ++i)
  //      {
  //        CvPoint *p = (CvPoint*)cvGetSeqElem( result, i );
  //        drawLine(contoursImg, Point2D<int>(p1->x, p1->y), Point2D<int>(p->x, p->y),
  //            PixRGB<byte>(255, 0, 0));
  //        p1 = p;

  //        //cvSeqPush(squares,
  //        //    (CvPoint*)cvGetSeqElem( result, i ));
  //      }
  //      //  LINFO("area=%f, mincos=%f", area, s);
  //    }
  //  }

  //  // take the next contour
  //  contours = contours->h_next;
  //}
  ////SHOWIMG(contoursImg);
  //
  //

  ////CvMemStorage* storage = cvCreateMemStorage(0);
  ////CvSeq* contours = 0;
  ////int  comp_count = 0;


  ////cvFindContours( marker_mask, storage, &contours, sizeof(CvContour),
  ////    CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
  ////IplImage* markers = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
  ////cvZero( markers );
  ////for( ; contours != 0; contours = contours->h_next, comp_count++ )
  ////{
  ////  cvDrawContours( markers, contours, cvScalarAll(comp_count+1),
  ////      cvScalarAll(comp_count+1), -1, -1, 8, cvPoint(0,0) );
  ////}
  ////SHOWIMG(ipl2float(markers));

  ////CvMat* color_tab = cvCreateMat( 1, comp_count, CV_8UC3 );
  ////CvRNG rng = cvRNG(-1);
  ////for(int i = 0; i < comp_count; i++ )
  ////{
  ////  uchar* ptr = color_tab->data.ptr + i*3;
  ////  ptr[0] = (uchar)(cvRandInt(&rng)%180 + 50);
  ////  ptr[1] = (uchar)(cvRandInt(&rng)%180 + 50);
  ////  ptr[2] = (uchar)(cvRandInt(&rng)%180 + 50);
  ////}

  ////{
  ////  double t = (double)cvGetTickCount();
  ////  cvWatershed( img2ipl(itsSMapCellsInput), markers );
  ////  t = (double)cvGetTickCount() - t;
  ////  LINFO( "exec time = %gms", t/(cvGetTickFrequency()*1000.) );
  ////}

  ////// paint the watershed image
  ////IplImage* wshed = cvCloneImage(img2ipl(itsSMapCellsInput));
  ////for(int i = 0; i < markers->height; i++ )
  ////  for(int j = 0; j < markers->width; j++ )
  ////  {
  ////    int idx = CV_IMAGE_ELEM( markers, int, i, j );
  ////    uchar* dst = &CV_IMAGE_ELEM( wshed, uchar, i, j*3 );
  ////    if( idx == -1 )
  ////      dst[0] = dst[1] = dst[2] = (uchar)255;
  ////    else if( idx <= 0 || idx > comp_count )
  ////      dst[0] = dst[1] = dst[2] = (uchar)0; // should not get here
  ////    else
  ////    {
  ////      uchar* ptr = color_tab->data.ptr + (idx-1)*3;
  ////      dst[0] = ptr[0]; dst[1] = ptr[1]; dst[2] = ptr[2];
  ////    }
  ////  }

  //////cvAddWeighted( wshed, 0.5, img_gray, 0.5, 0, wshed );
  ////SHOWIMG(ipl2rgb(wshed));




  itsEvc->input(itsSMapCellsInput);
  //itsEvc->input(dst);
  itsSMap =  itsEvc->getVCXmap();

  //Image<float> lum = luminance(dst);
  Image<float> lum = luminance(itsSMapCellsInput);
  inplaceNormalize(lum, 0.0F, 255.0F);

  SMapState background;
  std::vector<SMapState> foregroundObjs;

  //Get the background/foreground region by looking at low saliency values
  Image<float> smap = rescale(itsSMap, itsSMapCellsInput.getDims());
  for(int y=0; y<smap.getHeight(); y++)
    for(int x=0; x<smap.getWidth(); x++)
    {
      Point2D<int> pixLoc(x,y);
      float val = smap.getVal(pixLoc);


      //Check if we have a foreground or background by looking at saliency values
      //and calculate the mean and variance of the pixels in the background/foreground
      float pixVal = lum.getVal(x,y);
      if (val < 40)
      {
        background.region.push_back(pixLoc);

        if (background.mu == -1)
        {
          background.mu = pixVal;
          background.sigma = 0;
        }
        else
        {
          //compute the stddev and mean of each feature
          //This algorithm is due to Knuth (The Art of Computer Programming, volume 2:
          //  Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.)

          //Mean
          const double delta = pixVal - background.mu;
          background.mu += delta/background.region.size();

          //variance
          if (background.region.size() > 2) //to avoid divide by 0
          {
            background.sigma = (background.sigma*(background.region.size()-2)) + delta*(pixVal-background.mu);
            background.sigma /= double(background.region.size()-1);
          }
        }

      } else {

        //find the foreground object closest to this pix in value and distance
        int objId = -1;
        double maxProb = 0.0;
        for(uint i=0; i<foregroundObjs.size(); i++)
        {
          double valProb = gauss<double>(pixVal, foregroundObjs[i].mu, foregroundObjs[i].sigma);
          //double distProb = gauss<double>(pixLoc.distance(foregroundObjs[i].center), 0,
          //    foregroundObjs[i].sigma);

          double prob = valProb; // + log(distProb);

          if (prob > maxProb && prob > 1.0e-3) //get the max prob and the prob with values grater then min prob
          {
            maxProb = prob;
            objId = i;
          }
        }

        if (objId == -1)
        {
          //New object add it
          SMapState objectRegion;
          objectRegion.mu = pixVal;
          objectRegion.sigma = 5;
          objectRegion.center = pixLoc;
          objectRegion.region.push_back(pixLoc);
          foregroundObjs.push_back(objectRegion);
        } else {
          //Add it to the exsisting object

          foregroundObjs[objId].region.push_back(pixLoc);

          //compute the stddev and mean of each feature
          //This algorithm is due to Knuth (The Art of Computer Programming, volume 2:
          //  Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.)


          //Mean
          const double delta = pixVal - foregroundObjs[objId].mu;
          foregroundObjs[objId].mu += delta/foregroundObjs[objId].region.size();

          //variance
          if (foregroundObjs[objId].region.size() > 2) //to avoid divide by 0
          {
            foregroundObjs[objId].sigma = (foregroundObjs[objId].sigma*(foregroundObjs[objId].region.size()-2)) + delta*(pixVal-foregroundObjs[objId].mu);
            foregroundObjs[objId].sigma /= double(foregroundObjs[objId].region.size()-1);
          }
        }
      }
    }

  //Add the objects to the map
  itsSMapState.clear();

  background.sigma = sqrt(background.sigma);
  itsSMapState.push_back(background);

  for(uint i=0; i<foregroundObjs.size(); i++)
  {
    foregroundObjs[i].sigma = sqrt(foregroundObjs[i].sigma);
    itsSMapState.push_back(foregroundObjs[i]);
  }

}


Layout<PixRGB<byte> > SMap::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  Image<PixRGB<byte> > in = itsSMapCellsInput;
  //inplaceNormalize(in, 0.0F, 255.0F);

  Image<float> perc(in.getDims(), ZEROS);

  for(uint i=0; i<itsSMapState.size(); i++)
  {
    SMapState rs = itsSMapState[i];
    for(uint pix=0; pix<rs.region.size(); pix++)
    {
      //Could sample from the disterbusion and show these pixels
      perc.setVal(rs.region[pix], rs.mu);
    }
  }

  Image<PixRGB<byte> > tmp = perc;
//  for(int y=0; y < tmpEdges.getHeight(); y++)
//    for(int x=0; x < tmpEdges.getWidth(); x++)
//      if (tmpEdges.getVal(x,y) > 0)
//        tmp.setVal(x,y,PixRGB<byte>(255,0,0));
//

  Image<byte> smap = rescale(itsSMap, in.getDims());
  inplaceNormalize(smap, (byte)0, (byte)255);

//   outDisp = hcat(in, toRGB(smap));
   outDisp = hcat(toRGB(smap), tmp); //toRGB(Image<byte>(perc)));

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

