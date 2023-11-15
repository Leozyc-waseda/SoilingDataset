/*!@file BeoSub/ComplexObject.C Simple shape models */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Kevin Jones <kevinjon@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/ComplexObject.C $
// $Id: ComplexObject.C 10746 2009-02-03 07:09:00Z itti $
//

#include "ComplexObject.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"
#include "GUI/XWindow.H"

#include <cmath>
#include <fstream>

// ######################################################################
ComplexObject::ComplexObject(char* name, char* vodbName)
{
  itsName = name;
  itsVODBName = vodbName;
  MYLOGVERB=LOG_CRIT;

  if (itsVODB.loadFrom(vodbName) == false)
    LFATAL("Cannot operate without a valid database.");



  std::string objectfilestr = vodbName;


  std::string filepathstr = objectfilestr.substr(0, objectfilestr.find_last_of('/')+1);


}

// ######################################################################

ComplexObject::ComplexObject(char* objFile)
{


  std::ifstream is(objFile);
  std::string name ="";
  std::string vodbName ="";
  std::string objFileStr = objFile;

  if (is.is_open() == false) {
    LERROR("Cannot open object '%s' ", objFile);
    return;
  }


  std::string filepath = objFileStr.substr(0, objFileStr.find_last_of('/')+1);

  getline(is, name);
  getline(is, vodbName);
  vodbName = filepath + vodbName;


 itsName = name;

  printf("VODB: %s \t Object Name: %s \n", vodbName.c_str(), itsName.c_str());
  itsVODBName = (char*)vodbName.c_str();
  MYLOGVERB=LOG_CRIT;

  if (itsVODB.loadFrom(vodbName) == false)
    LFATAL("Cannot operate without a valid database.");



  std::string objectfilestr = vodbName;


  std::string filepathstr = objectfilestr.substr(0, objectfilestr.find_last_of('/')+1);

}


// ######################################################################
ComplexObject::~ComplexObject()
{
}



// ######################################################################
int ComplexObject::matchKeypoints(bool showAll, Image< PixRGB<byte> > inputImg, VisualObjectMatchAlgo voma, std::vector < rutz::shared_ptr<VisualObjectMatch> >& matches, Image< PixRGB<byte> >& kpImg, Image< PixRGB<byte> >& fusedImg)
{



  rutz::shared_ptr<VisualObject>
    vo(new VisualObject("mypic", "mypicfilename", inputImg,
                        Point2D<int>(-1,-1), std::vector<float>(),
                        std::vector< rutz::shared_ptr<Keypoint> >(), true));
  return matchKeypoints(showAll, vo, voma, matches, kpImg, fusedImg);

}


// ######################################################################
int ComplexObject::matchKeypoints(bool showAll, rutz::shared_ptr<VisualObject> vo, VisualObjectMatchAlgo voma, std::vector < rutz::shared_ptr<VisualObjectMatch> >& matches, Image< PixRGB<byte> >& kpImg, Image< PixRGB<byte> >& fusedImg)
{


  // create visual object and extract keypoints:
  //  rutz::shared_ptr<VisualObject> vo(new VisualObject("mypic", "mypicfilename", inputImg));


  // get the matching objects:
  const uint nmatches = itsVODB.getObjectMatches(vo, matches, voma, 1);


  // prepare the fused image:
  Image< PixRGB<byte> > mimg;
  std::vector<Point2D<int> > tl, tr, br, bl;



 // if no match, forget it:
  if (nmatches == 0U)
    {
          printf("### No matching object found.\n");
    }
  else
    {

      // let the user know about the matches:
      //NOTE: while this doesn't loop through all found macthes now, it WILL still find all macthes, even if the user doesn't want to show them all. This is slow, and should be FIXed
      if(showAll) {
        for (uint i = 0; i < nmatches; i ++)
          {
            rutz::shared_ptr<VisualObjectMatch> vom = matches[i];
            rutz::shared_ptr<VisualObject> obj = vom->getVoTest();

            printf("### %s Object match with '%s' score=%f\n",
                   itsName.c_str(), obj->getName().c_str(), vom->getScore());

            // add to our fused image if desired:



            //we should probably test to see if the user passed a fusedImg in by reference before fusing the images together here (it'd save some time)
            mimg = vom->getTransfTestImage(mimg);

            // also keep track of the corners of the test image, for
            // later drawing:
            Point2D<int> ptl, ptr, pbr, pbl;
            vom->getTransfTestOutline(ptl, ptr, pbr, pbl);
            tl.push_back(ptl); tr.push_back(ptr);
            br.push_back(pbr); bl.push_back(pbl);

          }
      }
      else {

        rutz::shared_ptr<VisualObjectMatch> vom = matches[0];
        if (vom->checkSIFTaffine() == false)//if affine is too wierd
          return 0;

        rutz::shared_ptr<VisualObject> obj = vom->getVoTest();
        kpImg = vom->getMatchImage(1.0F);//NOTE: right now, only works if One match is to be shown!
        printf("### %s Object match with '%s' score=%f\n",
               itsName.c_str(), obj->getName().c_str(), vom->getScore());
      }

       // do a final mix between given image and matches:
      /*
        mimg = Image<PixRGB<byte> >(mimg * 0.5F + inputImg * 0.5F);

        // finally draw all the object outlines:
        PixRGB<byte> col(255, 255, 0);
        for (uint i = 0; i < tl.size(); i ++)
          {
            drawLine(mimg, tl[i], tr[i], col, 1);
            drawLine(mimg, tr[i], br[i], col, 1);
            drawLine(mimg, br[i], bl[i], col, 1);
            drawLine(mimg, bl[i], tl[i], col, 1);
          }

        fusedImg = mimg;
      */
    }


  return nmatches;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
