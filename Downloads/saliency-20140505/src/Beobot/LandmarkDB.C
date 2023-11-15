/*!@file Beobot/LandmarkDB.C manages groups of landmarks, which includes
  spatial,geographical, temporal and episodic information               */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/LandmarkDB.C $
// $Id: LandmarkDB.C 15495 2014-01-23 02:32:14Z itti $
//

#include "Beobot/LandmarkDB.H"
#include "Image/DrawOps.H"      // for drawing
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Util/Timer.H"

#include <cstdio>

// ######################################################################
LandmarkDB::LandmarkDB(uint nSegment):
  itsScenes(),
  itsLandmarks(),
  itsActiveLmkList(),
  itsSkipList(),
  itsSessionNames(),
  itsSessionLandmarks()
{
  if(nSegment > 0) resize(nSegment);
}

// ######################################################################
LandmarkDB::~LandmarkDB()
{ }

// ######################################################################
void LandmarkDB::build
( std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
  std::vector<Point2D<int> > &objOffset, uint fNum, uint currSegNum,
  rutz::shared_ptr<VisualObject> scene)
{
  ASSERT(currSegNum < itsLandmarks.size());
  LINFO("BEFORE lmks[%d].sz: %" ZU , currSegNum,
        itsLandmarks[currSegNum].size());

  // set the full scene size
  itsScenes.push_back(scene);
  int imgW   = scene->getImage().getWidth();
  int imgH   = scene->getImage().getHeight();
  float imgD = sqrt(imgW*imgW + imgH*imgH);
  Dims sDims = scene->getImage().getDims();

  // make sure only to consider object with > 5 keypoints
  uint onobj = inputVO.size();
  kpFilter(inputVO, objOffset);

  // for each objects get the matching scores
  // consider both the quality of match and object distance
  uint nobj = inputVO.size(); if(nobj == 0) return;
  uint nlmk = itsLandmarks[currSegNum].size();
  LINFO("get scores: num objects: %d, num landmarks: %d", nobj, nlmk);
  std::vector<std::vector<rutz::shared_ptr<VisualObjectMatch> > >
    resMatch(nobj);
  std::vector<std::vector<bool> > inDB(nobj);
  std::vector<std::vector<bool> > inTDB(nobj);
  std::vector<std::vector<int> > tIndex(nobj);
  std::vector<std::vector<float> > scores(nobj);
  std::vector<std::vector<float> > siftscores(nobj);
  std::vector<std::vector<float> > sdiffscores(nobj);
  for(uint i = 0; i < nobj; i++)
    {
      // check the salient region against existing landmarks
      for(uint j = 0; j < nlmk; j++)
        {
          if(itsActiveLmkList[currSegNum][j])
            {
              bool indb; bool intdb; int tindex;
              rutz::shared_ptr<VisualObjectMatch> cmatch =
                itsLandmarks[currSegNum][j]
                ->buildCheck(inputVO[i], objOffset[i], fNum,
                             indb, intdb, tindex);

              LINFO("[%d] Match obj[%d] with lmk[%d][%d]: {%d %d %d}",
                    fNum, i, currSegNum, j, indb, intdb, tindex);

              inDB[i].push_back(indb);
              inTDB[i].push_back(intdb);
              tIndex[i].push_back(tindex);
              resMatch[i].push_back(cmatch);

              // get the sift score
              float siftscore = 0.0f;
              if(indb || intdb) siftscore = cmatch->getScore();

              // get offset distance
              float dscore = 0.0;
              if(indb || intdb)
                dscore = getOffsetDistScore
                  (itsLandmarks[currSegNum][j], indb, intdb, tindex,
                   inputVO[i], objOffset[i], sDims, cmatch);

              // get overlap score
              float oscore = 0.0;
              if(indb || intdb) oscore = getOverlapScore(cmatch);

              // get the salient-based score
              float sdist = 0.0f; float salscore = 0.0f;
              float sdiff = cmatch->getSalDiff();
              if(indb || intdb)
                {
                  sdist = cmatch->getSalDist();
                  sdist = 1.0 - sdist/imgD;
                  salscore = sdist * sdiff;
                }

              float score = salscore;
              scores[i].push_back(score);
              siftscores[i].push_back(siftscore);
              sdiffscores[i].push_back(sdiff);
              LINFO("match obj[%d] & lmks[%d][%d] sift[%f %f %f] sal[%f %f]",
                    i, currSegNum, j, siftscore, dscore, oscore,
                    sdist, sdiff);
              // FIX: && (fNum%20 == 0)
              //if((indb || intdb)) Raster::waitForKey();
            }
          else
            {
              //LINFO("landmark[%d][%d] is inactive", currSegNum,j);
              inDB[i].push_back(false);
              inTDB[i].push_back(false);
              tIndex[i].push_back(-1);
              resMatch[i].push_back(rutz::shared_ptr<VisualObjectMatch>());
              scores[i].push_back(0.0f);
              siftscores[i].push_back(0.0f);
              sdiffscores[i].push_back(0.0f);
            }
          //Raster::waitForKey();
        }
    }

  // second pass:
  // check all the objects that is still not a candidate for any landmarks
  // usually because the SIFT module reject it.
  // we will now try to indirectly estimate the correct landmark
  for(uint i = 0; i < nobj; i++)
    {
      bool matched = false;
      // check if this object is matched
      for(uint j = 0; j < nlmk; j++) if(scores[i][j] > 0.0) matched = true;

      // check the salient region against existing landmarks
      if(!matched)
        {
          LINFO("object %d not matched", i);

          for(uint j = 0; j < nlmk; j++)
            {
              if(itsActiveLmkList[currSegNum][j])
                {
                  // get the latest frame of that landmark
                  bool indb; bool intdb; int tindex = -1;
                  rutz::shared_ptr<VisualObject> lobj
                    = itsLandmarks[currSegNum][j]->getLatestObject
                    (indb, intdb, tindex);

                  LINFO("got latest object of lmk[][%d]: {%d %d %d}", 
                        j, indb, intdb, tindex);

                  const std::vector<float>& feat1 = lobj->getFeatures();
                  const std::vector<float>& feat2 = inputVO[i]->getFeatures();

                  // feature similarity  [ 0.0 ... 1.0 ]:
                  float cval = 0.0;
                  for(uint k = 0; k < feat1.size(); k++)
                    {
                      float val = pow(feat1[k] - feat2[k], 2.0); cval += val;
                    }
                  cval = sqrt(cval/feat1.size());
                  float sscore =  1.0F - cval;

                  if(sscore > .80)
                    {
                      LINFO("obj[%d] lmk[%d] good match (%f) check SIFT",
                            i, j, sscore);

                      uint lfNum =
                        itsLandmarks[currSegNum][j]->getLatestFNum();

                      // match the two scenes
                      LINFO("matching scenes: %d and %d", lfNum, fNum);
                      rutz::shared_ptr<VisualObject> a = itsScenes[lfNum];
                      rutz::shared_ptr<VisualObject> b = itsScenes[fNum];

                      // check SIFT match
                      float kpAvgDist, afAvgDist, score;
                      bool isSIFTaffine; SIFTaffine siftAffine;
                      VisualObjectMatchAlgo voma(VOMA_SIMPLE);

                      // compute the matching keypoints:
                      Timer tim(1000000); tim.reset();
                      rutz::shared_ptr<VisualObjectMatch>
                        matchRes(new VisualObjectMatch(a, b, voma));
                      uint64 t = tim.get();

                      // let's prune the matches:
                      uint orgSize = matchRes->size();
                      tim.reset();
                      uint np = matchRes->prune();
                      uint t2 = tim.get();

                      LINFO("Found %u matches (%s & %s) in %.3fms:"
                            " pruned %u in %.3fms",
                            orgSize, a->getName().c_str(),
                            b->getName().c_str(), float(t) * 0.001F,
                            np, float(t2) * 0.001F);

                      // matching score
                      kpAvgDist    = matchRes->getKeypointAvgDist();
                      afAvgDist    = matchRes->getAffineAvgDist();
                      score        = matchRes->getScore();
                      isSIFTaffine = matchRes->checkSIFTaffine
                        (M_PI/4,5.0F,0.25F);
                      siftAffine   = matchRes->getSIFTaffine();
                      LINFO("kpAvgDist = %.4f|affAvgDist = %.4f|"
                            " score: %.4f|aff? %d",
                            kpAvgDist, afAvgDist, score, isSIFTaffine);

                      if (!isSIFTaffine)
                        LINFO("### Affine is too weird -- BOGUS MATCH");
                      else
                        {
                          // show our final affine transform:
                          LINFO("[testX]  [ %- .3f %- .3f ] [refX]   [%- .3f]",
                                siftAffine.m1, siftAffine.m2, siftAffine.tx);
                          LINFO("[testY]= [ %- .3f %- .3f ] [refY] + [%- .3f]",
                                siftAffine.m3, siftAffine.m4, siftAffine.ty);
                        }

                      bool isSIFTfit = (isSIFTaffine && (score > 2.5) &&
                                        (matchRes->size() > 3));
                      LINFO("OD isSIFTfit %d", isSIFTfit);

                      if(isSIFTfit)
                        {
                          // get the location
                          Point2D<int> loffset = itsLandmarks[currSegNum][j]
                            ->getLatestOffsetCoords();
                          Point2D<int> salpt1 = lobj->getSalPoint() + loffset;
                          Point2D<int> salpt2 = inputVO[i]->getSalPoint() +
                            objOffset[i];
                          LINFO("loffset: %d %d", loffset.i, loffset.j);

                          // forward affine transform [A * ref -> tst ]
                          float u, v; siftAffine.transform
                                        (salpt1.i, salpt1.j, u, v);
                          float dist = salpt2.distance
                            (Point2D<int>(int(u+0.5F), int(v+0.5F)));
                          LINFO("pos1: (%d,%d) -> (%f,%f) & "
                                "pos2: (%d,%d): dist: %f",
                                salpt1.i, salpt1.j, u, v,
                                salpt2.i, salpt2.j, dist);

                          float sdist  = 1.0 - dist/imgD;

                          if(dist < 10.0F)
                            {
                              inDB[i][j] = indb;
                              inTDB[i][j] = intdb;
                              tIndex[i][j] = tindex;
                              scores[i][j] = (sdist * sscore);
                              siftscores[i][j] = (score);
                              sdiffscores[i][j] = (sscore);                             
                            }
                        }
                    }
                  else{ LINFO("score too low: %f",sscore); }
                }
            }
        }
    }

  // check if there are more than 1 objects being properly matched

  // FIX: if there are landmarks that can match with more than 1 objects.
  // we can only keep the best one
  // that other one hopefully would be closer to another landmark
  // else it will be a new landmark by itself

//   printScores(inputVO, currSegNum, siftscores);
//   printScores(inputVO, currSegNum, sdiffscores);
//   printScores(inputVO, currSegNum, scores);

  LINFO("skipping: ");
  for(int i = 0; i < int(onobj - nobj); i++)
    LINFO("%d %s", i, itsSkipList[itsSkipList.size()-1-i]->getName().c_str());
  //Raster::waitForKey();

  // while there is an insertable object
  std::vector<bool> inserted(nobj);
  std::vector<bool> lmatched(nlmk);
  for(uint i = 0; i < nobj; i++) inserted[i] = false;
  for(uint j = 0; j < nlmk; j++) lmatched[j] = false;

  // for each object
  std::vector<float> ratio(nobj);
  for(uint i = 0; i < nobj; i++) ratio[i] = 0.0;
  bool done = false; if(nlmk == 0) done = true;
  while(!done)
    {
      // calculate the best/2nd best match ratio
      float mratio; int mind; std::vector<int> mlist;
      findBestMatch(scores, ratio, inserted, lmatched, mratio, mind, mlist);

      // if no more matches, we are done
      if(mind == -1) done = true;
      else
        {
          int ilbest = mlist[0];
          // for multiple matches
          if(mlist.size() > 1)
            {
              LINFO("transfer");
              //Raster::waitForKey();
              // we want to create momentum to 1 landmark

              // FIX: if the best ratio is not much better than the others

              // find the best landmark
              // and insert all matching salient regions to it
              int ilmax = -1; int maxlsize = -1;
              for(uint j = 0; j < mlist.size(); j++)
                {
                  // check the landmark sizes
                  int clobjsize  = 
                    itsLandmarks[currSegNum][mlist[j]]->numObjects();
                  int cltobjsize = 
                    itsLandmarks[currSegNum][mlist[j]]->numTempObjects();
                  int clsize = clobjsize + cltobjsize; 

                  LINFO("lsize[%d]: %d + %d =  %d (%d)", mlist[j], 
                        clobjsize, cltobjsize, clsize, tIndex[mind][mlist[j]]);

                  if(clsize > maxlsize){ ilmax = mlist[j]; maxlsize = clsize; }
                }
              ilbest = ilmax;
              LINFO("ilbest: %d", ilbest);

              // if they differ by more than 2
              // else push the object to the latest one

              // insert one by one to the new landmark
              for(uint j = 0; j < mlist.size(); j++)
                {
                  // the matching frames from the other landmark
                  // is inserted first
                  if(mlist[j] != ilbest)
                    {
                      // check if this frame is already in the evidence
                      LINFO("transf lmk[][%d] to lmk[][%d] {%d} ", 
                            mlist[j], ilbest, tIndex[mind][mlist[j]]);

                      // transfer visual objects properly
                      itsLandmarks[currSegNum][ilbest]->transferEvidence
                        (itsLandmarks[currSegNum][mlist[j]],
                         inDB[mind][mlist[j]], inTDB[mind][mlist[j]],
                         tIndex[mind][mlist[j]], resMatch[mind][mlist[j]]);
                    }
                }
            }

          // insert inputVO to the best match ratio
          LINFO("Inserting obj %d to landmark[%d][%d] [%d --> %d %d]",
                mind, currSegNum, ilbest,  tIndex[mind][ilbest],
                itsLandmarks[currSegNum][ilbest]->numObjects(),
                itsLandmarks[currSegNum][ilbest]->numTempObjects());

          itsLandmarks[currSegNum][ilbest]
            ->build(inputVO[mind], objOffset[mind], fNum,
                    inDB[mind][ilbest], inTDB[mind][ilbest],
                    tIndex[mind][ilbest], resMatch[mind][ilbest]);

          // note object is already inserted (landmark matched)
          inserted[mind] = true;
          lmatched[mlist[0]] = true;
        }
    }

  // for the rest of the objects create new landmarks
  for(uint i = 0; i < nobj; i++)
    {
      if(!inserted[i])
        {
          LINFO("create landmark %" ZU , itsLandmarks[currSegNum].size());
          std::string
            lmName(sformat("landmark%07" ZU , itsLandmarks[currSegNum].size()));
          rutz::shared_ptr<Landmark> newlm
            (new Landmark(inputVO[i], objOffset[i], fNum, lmName));
          newlm->setMatchWin(itsWin);
          itsLandmarks[currSegNum].push_back(newlm);
          itsActiveLmkList[currSegNum].push_back(true);
        }
    }

  // check if there is a landmark with 0 size
  for(uint j = 0; j < nlmk; j++)
    {
      // check both landmark sizes
      int clsize =
        itsLandmarks[currSegNum][j]->numObjects() +
        itsLandmarks[currSegNum][j]->numTempObjects();
      // make it inactive
      if(clsize == 0) itsActiveLmkList[currSegNum][j] = false;
    }

  LINFO("AFTER lmks[%d].sz: %" ZU , currSegNum, itsLandmarks[currSegNum].size());

  // make landmarks that are not receiving frames the last NFDIFF inactive
  classifyInactiveLandmarks(fNum, NFDIFF);
}

// ######################################################################
void LandmarkDB::kpFilter
( std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
  std::vector<Point2D<int> > &objOffset)
{
  LINFO("  BEFORE kpFilter: %" ZU , inputVO.size());

  // while we still have a vo to check
  std::vector<rutz::shared_ptr<VisualObject> >::iterator
    voitr = inputVO.begin();
  std::vector<Point2D<int> >::iterator obitr = objOffset.begin();

  uint i = 0;
  while (voitr < inputVO.end())
    {
      // skip if we have less than 5 keypoints
      if((*voitr)->numKeypoints() <= 5)
        {
          LINFO("skip: %s (%d kp)", (*voitr)->getName().c_str(),
                (*voitr)->numKeypoints());
          itsSkipList.push_back((*voitr));

          voitr = inputVO.erase(voitr);
          obitr = objOffset.erase(obitr);
        }
      else{ voitr++; obitr++; }
      i++;
    }
  LINFO("  AFTER kpFilter: %" ZU , inputVO.size());
}

// ######################################################################
float LandmarkDB::getOffsetDistScore
( rutz::shared_ptr<Landmark> landmark, int indb, int intdb, int tindex,
  rutz::shared_ptr<VisualObject> vo, Point2D<int> offset, Dims sDims,
  rutz::shared_ptr<VisualObjectMatch> cmatch)
{
  float dscore = 0.0;

  Point2D<int> objOffset2;
  if(indb)
    objOffset2 = landmark->getOffsetCoords(tindex);
  else
    objOffset2 = landmark->getTempOffsetCoords(tindex);
  bool isODmatch = (vo == cmatch->getVoRef());
  Point2D<int> diff;
  if(isODmatch)
    {
      diff = cmatch->getSpatialDist(offset, objOffset2);
      //itsWin->drawImage(cmatch->getMatchImage(sDims, offset, objOffset2),0,0);
    }
  else
    {
      diff = cmatch->getSpatialDist(objOffset2, offset);
      //itsWin->drawImage(cmatch->getMatchImage(sDims, objOffset2, offset),0,0);
    }
  float dist = diff.distance(Point2D<int>(0,0));

  // dist: 0 to 25.0 -> range 1.0 to .98
  if(dist <= 25.0)
    dscore = 1.0 - (dist/25.0) * .02;
  // dist 25.0 to 200.0 -> range .98 to .01
  else if(dist > 25.0 && dist <= 200.0)
    dscore = .98 - ((dist - 25.0)/175.0)* .97;
  // dist > 200.0 -> range .01
  else
    dscore = .01;
  LINFO("dist: (%d,%d): %f", diff.i, diff.j, dist);

  return dscore;
}

// ######################################################################
float LandmarkDB::getOverlapScore
(rutz::shared_ptr<VisualObjectMatch> cmatch)
{
  float oscore = 0.0;

  int area1 = cmatch->getVoRef()->getImage().getSize();
  int area2 = cmatch->getVoTest()->getImage().getSize();
  Rectangle rovl = cmatch->getOverlapRect();

  if(!rovl.isValid()) return oscore;

  float ovl = float(rovl.width() * rovl.height());
  oscore = (ovl/area1 + ovl/area2)/2.0;
  LINFO("area1: %d, area2: %d, ovl: %f, oscore: %f",
        area1, area2, ovl, oscore);
  return oscore;
}

// ######################################################################
void LandmarkDB::printScores
( std::vector<rutz::shared_ptr<VisualObject> > inputVO,
  int currSegNum, std::vector<std::vector<float> > inscores)
{
  int printlimit = 25;
  uint nobj = inputVO.size();
  uint nlmk = itsActiveLmkList[currSegNum].size();
  LINFO("nobj: %d nlmk: %d", nobj, nlmk);

  // print sal scores
  printf("          ");
  int lcount = 0;
  for(uint j = 0; j < nlmk; j++)
    {
      if(itsActiveLmkList[currSegNum][j] && lcount < printlimit)
       {
         printf("%6d",j);
         lcount++;
       }
    }
  printf("\n");
  for(uint i = 0; i < nobj; i++)
    {
      int len = inputVO[i]->getName().length();
      printf("%10s", inputVO[i]->getName().substr(len-10).c_str());
      lcount = 0;
      for(uint j = 0; j < nlmk; j++)
        {
          if(itsActiveLmkList[currSegNum][j] && lcount < printlimit)
            {
              printf("%6.2f", inscores[i][j]);
              lcount++;
            }
        }
      printf("\n");
    }
  printf("\n");
}

// ######################################################################
void LandmarkDB::findBestMatch
(std::vector<std::vector<float> > scores, std::vector<float> &ratio,
 std::vector<bool> &inserted, std::vector<bool> &lmatched,
 float &mratio, int &mind, std::vector<int> &mlist)
{
  // calculate ratio for each object
  for(uint i = 0; i < scores.size(); i++)
    {
      ratio[i] = 0.0;
      // make sure at least 1 landmark is still available to insert to
      bool hasMatch = false;
      for(uint j = 0; j < scores[i].size(); j++)
        if(scores[i][j] > 0.0 && !lmatched[j]) hasMatch = true;

      // check if it inserted (or inactive)
      if(!inserted[i] && hasMatch)
        {
          // go through each landmark
          float max = 0.0f; float max2 = 0.0f;
          for(uint j = 0; j < scores[i].size(); j++)
            {
              if(scores[i][j] > max)
                { max2 = max; max = scores[i][j]; }
              else if(scores[i][j] > max2)
                { max2 = scores[i][j]; }
            }
          if(max2 < .25) ratio[i] = max/.25;
          else           ratio[i] = max/max2;
        }
      LINFO("ratio[%d]: %f", i, ratio[i]);
    }

  // go through each ratio
  mratio = 0.0; mind = -1;
  for(uint i = 0; i < ratio.size(); i++)
    {
      if(!inserted[i])
        {
          if(ratio[i] > mratio)
            { mratio = ratio[i]; mind = i; }
        }
    }
  if(mind == -1) { LINFO("no more matches"); return; }

  // get the matching indexes
  for(uint j = 0; j < scores[mind].size(); j++)
    {
      // check if it inserted (or inactive)
      if(!lmatched[j] && scores[mind][j] > 0.0)
        mlist.push_back(j);
    }
  LINFO("max ratio: %f obj[%d]: %" ZU " matches", mratio, mind, mlist.size());
}

// ######################################################################
void LandmarkDB::classifyInactiveLandmarks(uint fNum, uint nfDiff, bool print)
{
  //bool stop = false;
  for(uint i = 0; i < itsLandmarks.size(); i++)
    for(uint j = 0; j < itsLandmarks[i].size(); j++)
      {
        if(itsActiveLmkList[i][j] &&
           ((itsLandmarks[i][j]->getLatestFNum() + nfDiff) < fNum))
          {
            // move the latest object on the temp VO to the DB
            itsLandmarks[i][j]->moveLatestTempVisualObjectToDB();
            itsActiveLmkList[i][j] = false;
            //stop = true;
            if(print) LINFO("made landmark [%d][%d] inactive", i, j);
          }

        if(itsActiveLmkList[i][j] && print)
          LINFO("landmark[%3d %3d]   active: %5d,%5d: %3d + %3d = %3d", i, j,
                itsLandmarks[i][j]->getVisualObjectFNum(0),
                itsLandmarks[i][j]->getLatestFNum(),
                itsLandmarks[i][j]->numObjects(),
                itsLandmarks[i][j]->numTempObjects(),
                itsLandmarks[i][j]->numObjects() +
                itsLandmarks[i][j]->numTempObjects());
        else if(itsLandmarks[i][j]->numObjects() > 0 && print)
          LINFO("landmark[%3d %3d] inactive: %5d,%5d: %3d + %3d = %3d", i, j,
                itsLandmarks[i][j]->getVisualObjectFNum(0),
                itsLandmarks[i][j]->getLatestFNum(),
                itsLandmarks[i][j]->numObjects(),
                itsLandmarks[i][j]->numTempObjects(),
                itsLandmarks[i][j]->numObjects() +
                itsLandmarks[i][j]->numTempObjects());
        else if(print)
          LINFO("landmark[%3d %3d] inactive:    -1,   -1: %3d + %3d = %3d", i, j,
                itsLandmarks[i][j]->numObjects(),
                itsLandmarks[i][j]->numTempObjects(),
                itsLandmarks[i][j]->numObjects() +
                itsLandmarks[i][j]->numTempObjects());
      }
  //if(stop) Raster::waitForKey();
}

// ######################################################################
void LandmarkDB::finishBuild(uint rframe)
{
  // make all landmarks inactive and then prune them
  classifyInactiveLandmarks(rframe, 0, true);
  pruneLandmarks();

  // show the Visual Objects skipped
  //printSkiplist();
}

// ######################################################################
void LandmarkDB::pruneLandmarks()
{
  // go through all segments
  for(uint i = 0; i < itsLandmarks.size(); i++)
    {
      std::vector< rutz::shared_ptr<Landmark> >::iterator itr =
        itsLandmarks[i].begin();
      uint ct = 0; uint orgCt = 0;
      // go through all the landmarks in the segment
      while (itr < itsLandmarks[i].end())
        {
          uint numObjects  = (*itr)->numObjects();
          uint numTempObjects = (*itr)->numTempObjects();
          uint numTotal = numObjects + numTempObjects;

          int range = 0; int start = -1; int end = -1;
          if(numObjects > 0)
            {
              start = (*itr)->getVisualObjectFNum(0);
              end   = (*itr)->getLatestFNum();
              range = end - start;
            }

          // criterion:
          // allow very persistent salient, slightly ephemeral object:
          // range <= 20 frames but numTotal >= 8
          // allow less persistently salient, but consistently detected:
          // range > 20 & numTotal >= 5
          if (((range <= 20) && (numTotal >= 8)) ||
              ((range >  20) && (numTotal >= 5))    )
            {
              // change the name to keep order
              std::string lmName(sformat("landmark%07u", ct));
              LINFO("include: %s [%d %d: %d] (%d + %d = %d): %s",
                    (*itr)->getName().c_str(), start, end, range,
                    numObjects, numTempObjects, numTotal, lmName.c_str());
              (*itr)->setName(lmName);
              ++ itr;
              ct++;
            }
          else
            {
              LINFO("Landmark[%d][%d] is too small: [%d %d: %d] %d + %d = %d",
                    i, orgCt, start, end, range,
                    numObjects, numTempObjects, numTotal);
              // remove from the list
              itr = itsLandmarks[i].erase(itr);
            }
          orgCt++;
        }
    }
}

// ######################################################################
void LandmarkDB::display()
{
  // can only display if there is a window display
  if(itsWin.is_invalid()) { LINFO("no window display"); return; }

  Dims d = itsWin->getDims();
  int w = d.w(), h = d.h();
  int nSegment = itsLandmarks.size();

  // check each segment
  for(int  i = 0; i < nSegment; i ++)
    {
      // how many landmarks recovered for this segment
      int size = itsLandmarks[i].size();
      printf("segment [%4d] has: %d landmarks\n",i,size);
      for(int j = 0; j < size; j++)
        {
          // how many objects for each landmark
          int nObjects  = itsLandmarks[i][j]->numObjects();
          int nTObjects = itsLandmarks[i][j]->numTempObjects();
          printf("   lm[%4d][%4d] has %d vo + %d temp =  %d\n", i, j,
                nObjects,  nTObjects, nObjects + nTObjects);

          // display each
          for(int k = 0; k < nObjects; k++)
            {
              printf("image %4d: %s   ", k, itsLandmarks[i][j]
                    ->getObject(k)->getName().c_str());
              Image<PixRGB<byte> > tIma(w,h,ZEROS);

              Point2D<int> salpt =
                itsLandmarks[i][j]->getObject(k)->getSalPoint();
              Point2D<int> offset =
                itsLandmarks[i][j]->getOffsetCoords(k);

              inplacePaste
                (tIma, itsLandmarks[i][j]->getObject(k)
                 ->getKeypointImage(1.0F,0.0F), offset);
              drawDisk(tIma, salpt+offset, 3, PixRGB<byte>(255,255,0));

              itsWin->drawImage(tIma,0,0);
              Raster::waitForKey();
            }
        }
    }
}

// ######################################################################
void LandmarkDB::printSkiplist()
{
  Dims d = itsWin->getDims();
  int w = d.w(), h = d.h();

  // show the skiplist
  LINFO("skipping %" ZU " images", itsSkipList.size());
  for(uint  i = 0; i < itsSkipList.size(); i ++)
    {
      Image<PixRGB<byte> > tIma(w,h,ZEROS);
      inplacePaste
        (tIma,itsSkipList[i]->getKeypointImage(1.0F,0.0F), Point2D<int>(0, 0));
      itsWin->drawImage(tIma,0,0);
      LINFO("Image %d:  %s has %d kp", i,
            itsSkipList[i]->getName().c_str(), itsSkipList[i]->numKeypoints());
      Raster::waitForKey();
    }
}

// ######################################################################
void LandmarkDB::setSession(std::string sessionFName, bool sort)
{
  // open the session file
  FILE *fp;  char inLine[300];
  if((fp = fopen(sessionFName.c_str(),"rb")) == NULL)
    LFATAL("session file: %s not found", sessionFName.c_str());
  LINFO("session file name: %s",sessionFName.c_str());

  // open the file
  itsSessionNames.clear();
  itsSessionLength.clear();
  itsSessionGroundTruth.clear();
  itsSessionGroundTruthSegmentLength.clear();

  // go through each session
  int session_count = 0;
  while(fgets(inLine, 300, fp) != NULL)
  {
    // get the files in this category and ground truth
    char sName[200]; int cStart, cNum; uint segnum;
    char gtfilename[300];
    int ret = sscanf(inLine, "%s %d %d %d %s", 
                     sName, &cStart, &cNum, &segnum, gtfilename);
    LINFO("[%3d] %20s: %d (%d - %d): %s", 
          segnum, sName, cNum, cStart, cStart+cNum-1, gtfilename);

    itsSessionNames.push_back(std::string(sName));
    itsSessionLength.push_back(cNum);    

    itsSessionGroundTruth.push_back(std::vector<GroundTruth>());
    itsSessionGroundTruthSegmentLength.push_back(-1.0F);
 
    // if a ground truth file is listed
    if(ret > 3)
      {
        float seg_length = 0;

        itsSessionGroundTruth[session_count] = 
          getGroundTruth(std::string(gtfilename), segnum, seg_length);
        itsSessionGroundTruthSegmentLength[segnum] = seg_length;
      }
    session_count++;
  }
  fclose(fp);

  // if needed to resort the salient regions
  if(sort) sortLandmarks();

  // reset the session related information
  setSessionInfo();
}

// ######################################################################
std::vector<GroundTruth> LandmarkDB::getGroundTruth
(std::string gt_filename, uint segnum, float &seg_length)
{
  //! setup ground truth
  //! given an annotation file create a ground truth file
  std::vector<GroundTruth> ground_truth;

  // open the file
  LINFO("opening: %s", gt_filename.c_str());
  FILE *fp;  char inLine[200]; //char comment[200];
  if((fp = fopen(gt_filename.c_str(),"rb")) == NULL)
    {
      LINFO("%s not found", gt_filename.c_str());
      return ground_truth;
    }

  // read out and discard the first line
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed"); 
  
  // read the segment lengths
  std::vector<float> seg_lengths;
  while(fgets(inLine, 200, fp) != NULL)
    {
      int segnum; float dist;
      int ret = sscanf(inLine, "%d %f", &segnum, &dist);
      LINFO("RET: %d [%d %f]", ret, segnum, dist);

      seg_lengths.push_back(dist);

      // check if we are still in segment length section
      if(ret != 2) break;
    }

  if(segnum < seg_lengths.size()) seg_length = seg_lengths[segnum];
  else seg_length = -1;

  // read each line 
  while(fgets(inLine, 200, fp) != NULL)
    {
      // get the three information
      uint snum; char t_iname[200]; float ldist;
      sscanf(inLine, "%s %d %f", t_iname, &snum, &ldist);

      // get the frame number
      std::string iname(t_iname);
      int ldpos = iname.find_last_of('.');
      int lupos = iname.find_last_of('_');
      std::string tnum = iname.substr(lupos+1, ldpos - lupos - 1);
      //LINFO("fname: %s %s [%d %f]", t_iname, tnum.c_str(), snum, ldist);
      int fnum  = atoi(tnum.c_str());
      
      if(snum == segnum)
        ground_truth.push_back(GroundTruth(fnum, snum, ldist/seg_length));
    }
  
  // while(fgets(inLine, 200, fp) != NULL)
  //   {
  //     // get the three information, discard comments
  //     int snum; char t_iname[200]; float ldist;
  //     sscanf(inLine, "%s %d %f %s", t_iname, &snum, &ldist, comment);
  //     curr_seg = snum;

  //     // get the frame number
  //     std::string iname(t_iname);
  //     int ldpos = iname.find_last_of('.');
  //     int lupos = iname.find_last_of('_');
  //     std::string tnum = iname.substr(lupos+1, ldpos - lupos - 1);
  //     LINFO("fname: %s %s [%d %f]", t_iname, tnum.c_str(), snum, ldist);
  //     int fnum  = atoi(tnum.c_str());

  //     // if not at the start of a segment
  //     int start_fnum = last_fnum + 1; 
  //     if(ldist == 0.0)
  //       { start_fnum = fnum; last_fnum = fnum; last_ldist = ldist; } 

  //     float range = float(fnum-last_fnum);
  //     if(range == 0.0F) range = 1.0F;

  //     LINFO("start_fnum: %d last_fnum: %d fnum: %d", 
  //           start_fnum, last_fnum, fnum);

  //     // 
  //     if(curr_seg == segnum)
  //       for(int i = start_fnum; i <= fnum; i++)
  //         {
  //           float c_ldist = 
  //             float(i - last_fnum)/range*(ldist-last_ldist) + last_ldist;

  //           ground_truth.push_back
  //             (GroundTruth(i,curr_seg,c_ldist/seg_length));
  //         }

  //     last_fnum  = fnum;
  //     last_ldist = ldist;
  //   }

  return ground_truth;
}

// ######################################################################
void LandmarkDB::sortLandmarks()
{
  for(uint i = 0; i < itsLandmarks.size(); i++)
    for(uint j = 0; j < itsLandmarks[i].size(); j++)
      itsLandmarks[i][j]->sort(itsSessionNames);
}

// #####################################################################
void LandmarkDB::setSessionInfo()
{
  // set the session information for individual landmarks
  for(uint i = 0; i < itsLandmarks.size(); i++)
    for(uint j = 0; j < itsLandmarks[i].size(); j++)
      itsLandmarks[i][j]->setSessionInfo();

  // set a landmarks database for each session
  setSessionLandmarks();

  // set the location range for each landmark
  setLocationRange();
}

// ######################################################################
void LandmarkDB::setSessionLandmarks()
{
  LINFO("set");
  uint nses = itsSessionNames.size();
  if(nses == 0) return;

  uint nseg = itsLandmarks.size();
  itsSessionLandmarks.resize(nses);
  for(uint i = 0; i < nses; i++)
    {
      std::string session = itsSessionNames[i];
      LDEBUG("[%d] session: %s", i, session.c_str());

      // go through all the segments
      itsSessionLandmarks[i].resize(nseg);
      for(uint j = 0; j < nseg; j++)
        {
          uint nlmk = itsLandmarks[j].size();
          LDEBUG("itsLmk[%d]: %d", j, nlmk);
          for(uint k = 0; k < nlmk; k++)
            {
              LDEBUG("check: itsLmk[%d][%d] ", j, k);
              if(itsLandmarks[j][k]->haveSessionVO(session))
                {
                  itsSessionLandmarks[i][j].push_back
                    (itsLandmarks[j][k]);
                  LDEBUG("match");
                }
            }
        }
    }
}

// #####################################################################
void LandmarkDB::setLocationRange()
{
  LINFO("set");

  // store each landmark first & last occurance location
  uint nseg = itsLandmarks.size();
  itsLandmarkLocationRange.resize(nseg);
  for(uint i = 0; i < nseg; i++)
    {
      // go through all the landmarks
      uint nlmk = itsLandmarks[i].size();
      itsLandmarkLocationRange[i].resize(nlmk);
      for(uint j = 0; j < nlmk; j++)
        {
          uint nsession = itsLandmarks[i][j]->getNumSession();

          float mfltrav = 1.0;
          float mlltrav = 0.0;
          for(uint k = 0; k < nsession; k++)
            {
              // get the session index range
              std::pair<uint,uint> r =
                itsLandmarks[i][j]->getSessionIndexRange(k);

              float fltrav = getLenTrav(i, j, r.first);
              float lltrav = getLenTrav(i, j, r.second);

              if(mfltrav > fltrav) mfltrav = fltrav;
              if(mlltrav < lltrav) mlltrav = lltrav;

              LDEBUG("[%d][%d]f-l: %f %f, m_f-l: %f %f",
                     i, j, fltrav, lltrav, mfltrav, mlltrav);
            }

          LDEBUG("m_f-l: %f %f", mfltrav, mlltrav);

          itsLandmarkLocationRange[i][j] =
            std::pair<float,float>(mfltrav, mlltrav);
        }
    }
}

// #####################################################################
float LandmarkDB::getLenTrav(uint snum, uint lnum, uint index)
{
  // get the object name
  std::string sname =
    itsLandmarks[snum][lnum]->getObject(index)->getName();
  LDEBUG("original sname %s", sname.c_str());
  sname = sname.substr(0, sname.find_last_of('_'));

  int lupos = sname.find_last_of('_');
  std::string tnum = sname.substr(lupos+1);
  int findex = atoi(tnum.c_str());

  sname = sname.substr(0, lupos);
  sname = sname.substr(0, sname.find_last_of('_'));

  uint i = 0;

  // check it against the names on the list
  while(i < itsSessionNames.size() && 
        sname.compare(itsSessionNames[i])) i++;

  // if found get the length traveled
  if(i < itsSessionNames.size())
    {
      if(itsSessionGroundTruth[i].size() > 0)
        {
          float ltrav = itsSessionGroundTruth[i][findex].ltrav;
          //LINFO("sname: %s findex: %d ltrav: %f * %f = %f ", 
          //      sname.c_str(), findex, ltrav,
          //      itsSessionGroundTruthSegmentLength[snum],
          //      ltrav*itsSessionGroundTruthSegmentLength[snum]);
          return ltrav;
        }
      else
        {
          float slen = float(itsSessionLength[i]);
          float fNum = float(itsLandmarks[snum][lnum]->
                             getVisualObjectFNum(index));
          LDEBUG("session name: %s: %f/%f = %f",
                 itsSessionNames[i].c_str(), fNum, slen, fNum/slen);
          return fNum/slen;
        }
    }
  else LFATAL("Session not in list: %s (%s)", sname.c_str(),
              itsLandmarks[snum][lnum]->getObject(index)->getName().c_str());
  return -1.0F;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
