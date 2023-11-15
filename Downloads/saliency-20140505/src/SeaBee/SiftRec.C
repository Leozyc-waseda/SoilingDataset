/*!@file SeaBee/SiftRec.C */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SiftRec.C $
// $Id: SiftRec.C 10794 2009-02-08 06:21:09Z itti $
//

#include "SeaBee/SiftRec.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Util/sformat.H"

const ModelOptionCateg MOC_SiftRecOps = {
    MOC_SORTPRI_3, "SiftRec options" };

const ModelOptionDef OPT_VDBFileName =
{ MODOPT_ARG(std::string), "VDBFileName", &MOC_SiftRecOps, OPTEXP_CORE,
    "The visual database file name",
    "vdb-filename", '\0', "", "objects.vdb" };

const ModelOptionDef OPT_SiftUseColor =
{ MODOPT_FLAG, "SiftUseColor", &MOC_SiftRecOps, OPTEXP_CORE,
    "Use color in the Sift descriptor",
    "use-color", '\0', "", "false" };


SiftRec::SiftRec(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName),
  itsVDBFile(&OPT_VDBFileName, this, ALLOW_ONLINE_CHANGES),
  itsUseColor(&OPT_SiftUseColor, this, ALLOW_ONLINE_CHANGES)
{
}

SiftRec::~SiftRec()
{
}


bool SiftRec::initVDB()
{
  itsTrainingMode = false;
  itsMaxLabelHistory = 1;
  itsVDB.loadFrom(itsVDBFile.getVal());

  return true;
}

void SiftRec::getObject(const Image<PixRGB<byte> > &img)
{
  ////Rescale the image to provide some scale inveriance
  //Image<PixRGB<byte> > inputImg = rescale(img, 256, 256);
  Image<PixRGB<byte> > inputImg = img;

  //float matchScore = 0;
  std::string objectName; //TODO = matchObject(inputImg, matchScore);
  if (objectName == "nomatch")
  {
    itsRecentLabels.resize(0);
    if (itsTrainingMode)
    {
      LINFO("Enter a lebel for this object:");
      std::getline(std::cin, objectName);
      if (objectName != "")
      {
        rutz::shared_ptr<VisualObject>
          vo(new VisualObject(objectName.c_str(), "NULL", inputImg,
                Point2D<int>(-1,-1),
                std::vector<double>(),
                std::vector< rutz::shared_ptr<Keypoint> >(),
                itsUseColor.getVal()));
        itsVDB.addObject(vo);
        itsVDB.saveTo(itsVDBFile.getVal());
      }
    }
  } else {
    itsRecentLabels.push_back(objectName);
    while(itsRecentLabels.size() > itsMaxLabelHistory)
      itsRecentLabels.pop_front();

    const std::string bestObjName = getBestLabel(1);

    if (bestObjName.size() > 0)
    {
      //itMsg->objectName = std::string("Test");
      //itMsg->objLoc.i = segInfo.rect.tl.i;
      //itMsg->objLoc.j = segInfo.rect.tl.j;

      //itMsg->confidence = matchScore*100.0F;
      //itsEventsPub->evolve(itMsg);
    }
  }

}

void SiftRec::trainObject(const Image<PixRGB<byte> > &img, const std::string& objectName)
{
  if (objectName != "")
  {
    rutz::shared_ptr<VisualObject>
      vo(new VisualObject(objectName.c_str(), "NULL", img,
            Point2D<int>(-1,-1),
            std::vector<double>(),
            std::vector< rutz::shared_ptr<Keypoint> >(),
            itsUseColor.getVal()));
    itsVDB.addObject(vo, false);
    itsVDB.saveTo(itsVDBFile.getVal());
  }
}

std::string SiftRec::getBestLabel(const size_t mincount)
{
  if (itsRecentLabels.size() == 0)
    return std::string();

  std::map<std::string, size_t> counts;

  size_t bestcount = 0;
  size_t bestpos = 0;

  for (size_t i = 0; i < itsRecentLabels.size(); ++i)
    {
      const size_t c = ++(counts[itsRecentLabels[i]]);

      if (c >= bestcount)
        {
          bestcount = c;
          bestpos = i;
        }
    }

  if (bestcount >= mincount)
    return itsRecentLabels[bestpos];

  return std::string();
}

std::string SiftRec::matchObject(Image<PixRGB<byte> > &ima, float &score, Rectangle& rect)
{
  //find object in the database
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
  rutz::shared_ptr<VisualObject>
    vo(new VisualObject("PIC", "PIC", ima,
          Point2D<int>(-1,-1),
          std::vector<double>(),
          std::vector< rutz::shared_ptr<Keypoint> >(),
          itsUseColor.getVal()));

  const uint nmatches = itsVDB.getObjectMatches(vo, matches, VOMA_SIMPLE,
      100U, //max objs to return
      0.5F, //keypoint distance score default 0.5F
      0.5F, //affine distance score default 0.5F
      1.0F, //minscore  default 1.0F
      3U, //min # of keypoint match
      100U, //keypoint selection thershold
      false //sort by preattentive
      );

  score = 0;
  float avgScore = 0, affineAvgDist = 0;
  int nkeyp = 0;
  int objId = -1;
  if (nmatches > 0)
  {
    rutz::shared_ptr<VisualObject> obj; //so we will have a ref to the last matches obj
    rutz::shared_ptr<VisualObjectMatch> vom;
    //for(unsigned int i=0; i< nmatches; i++){
    for (unsigned int i = 0; i < 1; ++i)
    {
      vom = matches[i];
      obj = vom->getVoTest();
      score = vom->getScore();
      nkeyp = vom->size();
      avgScore = vom->getKeypointAvgDist();
      affineAvgDist = vom->getAffineAvgDist();

      Point2D<int> tl, tr, br, bl;
      vom->getTransfTestOutline(tl, tr, br, bl);
      LINFO("%ix%i %ix%i %ix%i %ix%i",
          tl.i, tl.j, tr.i, tr.j,
          br.i, br.j, bl.i, bl.j);

      //if (tl.i+br.i, tl.j+br.j)
      //{
      //  Dims rectDims = Dims(tl.i+br.i, tl.j+br.j);
      //  rect = Rectangle(tl, rectDims); //Onlyh the tl and br
      //}


      objId = atoi(obj->getName().c_str()+3);

      return obj->getName();
      LINFO("### Object match with '%s' score=%f ID:%i",
          obj->getName().c_str(), vom->getScore(), objId);

      //calculate the actual distance (location of keypoints) between
      //keypoints. If the same patch was found, then the distance should
      //be close to 0
      double dist = 0;
      for (int keyp=0; keyp<nkeyp; keyp++)
      {
        const KeypointMatch kpm = vom->getKeypointMatch(keyp);

        float refX = kpm.refkp->getX();
        float refY = kpm.refkp->getY();

        float tstX = kpm.tstkp->getX();
        float tstY = kpm.tstkp->getY();
        dist += (refX-tstX) * (refX-tstX);
        dist += (refY-tstY) * (refY-tstY);
      }

      //   printf("%i:%s %i %f %i %f %f %f\n", objNum, obj->getName().c_str(),
      //       nmatches, score, nkeyp, avgScore, affineAvgDist, sqrt(dist));

      //analizeImage();
    }

  }

  return std::string("nomatch");
  }

