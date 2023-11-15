/*!@file FeatureMatching/DPM.C */


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
// $HeadURL$
// $Id$
//

#include "FeatureMatching/DPM.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/CutPaste.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/ShapeOps.H"
#include "SIFT/FeatureVector.H"
#include "GUI/DebugWin.H"
#include "Image/Layout.H"
#include "Image/Convolutions.H"




DPM::DPM() :
  itsInterval(10)
{
    itsThreadServer.reset(new WorkThreadServer("DPM", 10));
}


DPM::~DPM()
{
}

bool equalDetections(const DPM::Detection& a,const DPM::Detection& b)
{
  return (a.score == b.score && b.bb == b.bb);
}

bool cmpDetections(const DPM::Detection& a,const DPM::Detection& b)
{
  return (a.score > b.score);
}

void DPM::computeFeaturePyramid(const Image<PixRGB<byte> >& img)
{
  int numBin = 8; 

  int width = img.getWidth();
  int height = img.getHeight();
  double sc = pow(2,1.0F/itsInterval);
  int maxScale = 1 + floor(log(std::min(width,height)/(5*numBin))/log(sc));

  HOG hog;

  itsFeaturesPyramid.clear();
  itsFeaturesPyramid.resize(maxScale + itsInterval);


  for(int i=0; i<itsInterval; i++)
  {
    double scale = 1.0/pow(sc,i);
    int width = (int)round((float)img.getWidth()*scale);
    int height = (int)round((float)img.getHeight()*scale);
    Image<PixRGB<byte> > scaled = rescale(img, width, height);

    //First 2x interval
    itsFeaturesPyramid[i].features = hog.getFeatures(scaled, numBin/2);
    itsFeaturesPyramid[i].scale = 2*scale;
    itsFeaturesPyramid[i].bins = numBin/2;

    //second 2x interval
    itsFeaturesPyramid[i+itsInterval].features = hog.getFeatures(scaled, numBin);
    itsFeaturesPyramid[i+itsInterval].scale = scale;
    itsFeaturesPyramid[i+itsInterval].bins = numBin;

    //Remaining intervals
    for(int j= i+itsInterval; j < maxScale; j+=itsInterval)
    {
      Image<PixRGB<byte> > scaled2 = rescale(scaled,
          scaled.getWidth()/2, scaled.getHeight()/2);
      scaled = scaled2;

      itsFeaturesPyramid[j+itsInterval].features = hog.getFeatures(scaled, numBin);
      itsFeaturesPyramid[j+itsInterval].scale = 0.5*itsFeaturesPyramid[j].scale;
      itsFeaturesPyramid[j+itsInterval].bins = numBin;
    }
  }

  //Padd the pyramid
  int xpadd = 11;
  int ypadd = 6;

  for(uint i=0; i<itsFeaturesPyramid.size(); i++)
  {
    ImageSet<double>& feature = itsFeaturesPyramid[i].features;
    Dims dims(feature[0].getWidth() + (xpadd+1)*2,
              feature[0].getHeight() + (ypadd+1)*2);
    ImageSet<double> newFeature(feature.size(), dims, ZEROS);
    for(uint f=0; f<newFeature.size(); f++)
      inplacePaste(newFeature[f], feature[f], Point2D<int>(xpadd+1, ypadd+1));

    //Write the boundary occlusion in the last feature
    int fID = 31;
    for(int y=0; y<ypadd+1; y++)
      for(int x=0; x<newFeature[fID].getWidth(); x++)
        newFeature[fID].setVal(x,y, 1);

    for(int y=newFeature[fID].getHeight()-ypadd; y<newFeature[fID].getHeight(); y++)
      for(int x=0; x<newFeature[fID].getWidth(); x++)
        newFeature[fID].setVal(x,y, 1);

    for(int y=ypadd+1; y < newFeature[fID].getHeight()-ypadd; y++)
    {
      for(int x=0; x<xpadd+1; x++)
        newFeature[fID].setVal(x,y, 1);
      for(int x=newFeature[fID].getWidth()-xpadd; x< newFeature[fID].getWidth(); x++)
        newFeature[fID].setVal(x,y, 1);
    }

    itsFeaturesPyramid[i].features = newFeature;


  }

}

void DPM::convolveModel()
{
  
  itsModelScores.clear();
  for(uint level=itsInterval; level<itsFeaturesPyramid.size(); level++)
  {
    LINFO("Level %i", level);

    std::vector<rutz::shared_ptr<DPMJob> > itsJobs;

    for(size_t c=0; c<itsModel.components.size(); c++)
    {
      itsJobs.push_back(rutz::make_shared(new DPMJob(this, c, level)));
      itsThreadServer->enqueueJob(itsJobs.back());
    }

    //Wait for jobs to finish
    for(size_t i=0; i<itsJobs.size(); i++)
      itsJobs[i]->wait();

    //Get the max score and the component that has that score
    Image<double> maxScore;
    Image<int> maxComp;

    for(size_t i=0; i<itsJobs.size(); i++)
    {
      Image<double> score = itsJobs[i]->getScore();
    
      if (maxScore.initialized())
      {
        //Since the levels in the pyramid could be different (due to a difference in size of the rootFilter) 
        //Only add the smaller amount
        int w = std::min(score.getWidth(), maxScore.getWidth());
        int h = std::min(score.getHeight(), maxScore.getHeight());

        int maxScoreWidth = maxScore.getWidth();
        int scoreWidth = score.getWidth();

        Image<double>::const_iterator scorePtr = score.begin();
        Image<double>::iterator maxScorePtr = maxScore.beginw();
        for(int y=0; y<h; y++)
          for(int x=0; x<w; x++)
            if (scorePtr[y*scoreWidth + x] > maxScorePtr[y*maxScoreWidth + x])
            {
              maxScorePtr[y*maxScoreWidth + x] = scorePtr[y*scoreWidth + x];
              maxComp[y*maxScoreWidth + x] = itsJobs[i]->getComponent();
            }
      } else {
        maxScore = score;
        maxComp = Image<int>(maxScore.getDims(), NO_INIT);
        maxComp.clear(-1);
      }
    }
    itsModelScores.push_back(ModelScore(maxScore, maxComp, level));
  }

}

Image<double> DPM::convolveComponent(const int comp, const int level)
{
  ImageSet<double>& imgFeatures = itsFeaturesPyramid[level].features;

  //Convolve the root filter
  ModelComponent& component = itsModel.components[comp];
  ImageSet<double>& rootFeatures = component.rootFilter;
  Image<double> score = convolveFeatures(imgFeatures, rootFeatures);

  std::vector<double> deformation(4);
  deformation[0] = 1000;
  deformation[1] = 0;
  deformation[2] = 1000;
  deformation[3] = 0;

  Image<double> scoreDef = distanceTrans(score, deformation);
  score = scoreDef;
  int scoreW = score.getWidth();
  int scoreH = score.getHeight();

  score += component.offset;

  //Convolve the parts at a finer resolution (2x)
  for(size_t p=0; p<component.parts.size(); p++)
  {
    ImageSet<double>& partImgFeatures = itsFeaturesPyramid[level-itsInterval].features;
    ModelPart& part = component.parts[p];
    ImageSet<double>& partFeatures = part.features;
    Image<double> partScore = convolveFeatures(partImgFeatures, partFeatures);

    //Apply the deformation
    Image<double> defScore = distanceTrans(partScore, part.deformation);
    Point2D<float> anchor = part.anchor + Point2D<int>(1,1) - Point2D<int>(11,6); //Pyramid offset

    int defScoreW = defScore.getWidth();
    int defScoreH = defScore.getHeight();
    //Add the score to the rootFilter by shifting the position
    for(int y=0; y<scoreH; y++)
      for(int x=0; x<scoreW; x++)
      {
        int px = anchor.i + x*2;
        int py = anchor.j + y*2;
        if (px > 0 && px < defScoreW &&
            py > 0 && py < defScoreH)
          score[y*scoreW + x] += defScore[py*defScoreW + px];
      }

  }

  return score;
}

std::vector<DPM::Detection> DPM::getBoundingBoxes(const float thresh)
{
  std::vector<Detection> detections; //Scores of the detections

  for(uint i=0; i<itsModelScores.size(); i++)
  {
    //Find detection over the threshold
    Image<double>::const_iterator scorePtr = itsModelScores[i].score.begin();
    const int w = itsModelScores[i].score.getWidth();
    const int h = itsModelScores[i].score.getHeight();

    for(int y=0; y<h; y++)
      for(int x=0; x<w; x++)
      {
        if (scorePtr[y*w+x] > thresh)
        {
          int level = itsModelScores[i].level;
          float scale = (float)itsFeaturesPyramid[level].bins/itsFeaturesPyramid[level].scale;
          int comp = itsModelScores[i].component.getVal(x,y);
          int paddX = 11;
          int paddY = 6;
          if (comp >= 0)
          {
            ModelComponent& component = itsModel.components[comp];
            ImageSet<double>& rootFeatures = component.rootFilter;
            Dims size = rootFeatures[0].getDims();
            int x1 = (x - paddX)*scale+1;
            int y1 = (y - paddY)*scale+1;

            int x2 = x1 + size.w()*scale - 1;
            int y2 = y1 + size.h()*scale - 1;

            Rectangle rect = Rectangle::tlbrI(y1,x1, y2, x2);
            detections.push_back(Detection(rect, scorePtr[y*w+x], comp));
          }
        }
      }
  }

  return detections;

}

std::vector<DPM::Detection> DPM::filterDetections(const std::vector<Detection>& detections, const float overlap)
{

  std::vector<Detection> filteredDetections;

  //Non-maximum suppression. 
  //Greedily select high-scoring detections and skip detections 
  //that are significantly covered by a previously selected detection.

  //This alg still need to be verified for correctness.
  for(uint i=0; i<detections.size(); i++)
  {
    //See if we overlap with this detection
    bool isOverlap = false;
    for(uint j=0; j<filteredDetections.size(); j++)
    {
      if (detections[i].bb.getOverlapRatio(filteredDetections[j].bb) > overlap)
      {
        isOverlap = true;
        //If we overlap with this one, then check if this has a better score
        if (detections[i].score > filteredDetections[j].score)
            filteredDetections[j] = detections[i];
      }
    }

    if (!isOverlap)
      filteredDetections.push_back(detections[i]);
  }

  //Remove duplicates
  std::sort(filteredDetections.begin(), filteredDetections.end(), cmpDetections);
  std::vector<Detection>::iterator itr = 
    std::unique(filteredDetections.begin(), filteredDetections.end(), equalDetections); 
  filteredDetections.resize(itr - filteredDetections.begin());


  return filteredDetections;

}

void DPM::dtHelper(const Image<double>::const_iterator src,
                  Image<double>::iterator dst,
                  Image<int>::iterator ptr,
                  int step,
                  int s1, int s2, int d1, int d2,
                  double a, double b)
{
  
  if (d2 >= d1) //Check if we are out of bounds
  {
    int d = (d1+d2) >> 1;
    int s = s1;
    //Get the max value using the quadratic function while iterating from s1+1 to s2
    for (int p = s1+1; p <= s2; p++) 
    {
      if (src[s*step] - a*squareOf(d-s) - b*(d-s) < 
          src[p*step] - a*squareOf(d-p) - b*(d-p))
        s = p;
    }
    dst[d*step] = src[s*step] - a*squareOf(d-s) - b*(d-s);
    ptr[d*step] = s;

    //Iteratively call the next locations
    dtHelper(src, dst, ptr, step, s1, s, d1, d-1, a, b);
    dtHelper(src, dst, ptr, step, s, s2, d+1, d2, a, b);
  }
}

Image<double> DPM::distanceTrans(const Image<double>& score,
                                const std::vector<double>& deformation)
{
  double ax = deformation[0];
  double bx = deformation[1];
  double ay = deformation[2];
  double by = deformation[3];

  Image<double> defScore(score.getDims(), ZEROS);
  Image<int> scoreIx(score.getDims(), ZEROS);
  Image<int> scoreIy(score.getDims(), ZEROS);
  Image<int>::iterator ptrIx = scoreIx.beginw();
  Image<int>::iterator ptrIy = scoreIy.beginw();

  Image<double> tmpM(score.getDims(), ZEROS);
  Image<int> tmpIx(score.getDims(), ZEROS);
  Image<int> tmpIy(score.getDims(), ZEROS);

  const Image<double>::const_iterator src = score.begin();
  Image<double>::iterator dst = defScore.beginw();
  Image<double>::iterator ptrTmpM = tmpM.beginw();
  Image<int>::iterator ptrTmpIx = tmpIx.beginw();
  Image<int>::iterator ptrTmpIy = tmpIy.beginw();

  int w = score.getWidth();
  int h = score.getHeight();


  for(int x=0; x<w; x++)
    dtHelper(src+x, ptrTmpM+x, ptrTmpIx+x, w,
        0, h-1, 0, h-1, ay, by);

  for(int y=0; y<h; y++)
    dtHelper(ptrTmpM+y*w, dst+y*w, ptrTmpIy+y*w, 1,
        0, w-1, 0, w-1, ax, bx);

  for(int y=0; y<h; y++)
    for(int x=0; x<w; x++)
    {
      int p = y*w+x;
      ptrIy[p] = ptrTmpIy[p];
      ptrIx[p] = ptrTmpIx[y*w+ptrTmpIy[p]];
    }


  return defScore;

}

Image<double> DPM::convolveFeatures(const ImageSet<double>& imgFeatures, 
                                   const ImageSet<double>& filterFeatures)
{
  if (imgFeatures.size() == 0)
    return Image<double>();

  ASSERT(imgFeatures.size() == filterFeatures.size());

  //Compute size of output
  int w = imgFeatures[0].getWidth() - filterFeatures[0].getWidth() + 1;
  int h = imgFeatures[0].getHeight() - filterFeatures[0].getHeight() + 1;

  int filtWidth = filterFeatures[0].getWidth();
  int filtHeight = filterFeatures[0].getHeight();
  int srcWidth = imgFeatures[0].getWidth();

  Image<double> score(w,h, ZEROS);

  for(uint i=0; i<imgFeatures.size(); i++)
  {
    Image<double>::const_iterator srcPtr = imgFeatures[i].begin();
    Image<double>::const_iterator filtPtr = filterFeatures[i].begin();
    Image<double>::iterator dstPtr = score.beginw();

    for(int y=0; y<h; y++)
      for(int x=0; x<w; x++)
      {
        //Convolve the filter
        double val = 0;
        for(int yp = 0; yp < filtHeight; yp++)
          for(int xp = 0; xp < filtWidth; xp++)
          {
            val += srcPtr[(y+yp)*srcWidth + (x+xp)] * filtPtr[yp*filtWidth + xp];
          }

        *(dstPtr++) += val;
      }
  }

  return score;
}

Image<PixRGB<byte> > DPM::getModelImage()
{
  int lineLength = 20;

  Layout<PixRGB<byte> > modelImg;

  HOG hog;
  for(size_t c=0; c<itsModel.components.size(); c++)
  {

    ModelComponent& component = itsModel.components[c];
    Image<PixRGB<byte> > compImage =
      hog.getHistogramImage(component.rootFilter);
    compImage = rescale(compImage, compImage.getDims()*2);
    Image<PixRGB<byte> > partsImage = compImage;
    Image<PixRGB<byte> > defImage(compImage.getDims(), ZEROS);

    //Paste the parts
    for(size_t p=0; p<component.parts.size(); p++)
    {
      ModelPart& part = component.parts[p];
      Image<PixRGB<byte> > partImg =
        hog.getHistogramImage(part.features);

      //Paste into the root filter
      Point2D<int> topLeft = Point2D<int>(part.anchor*lineLength);
      inplacePaste(partsImage, partImg, topLeft);
      //Draw a border around the part
      drawRect(partsImage, Rectangle(topLeft, partImg.getDims()),
          PixRGB<byte>(255,0,0));

      //Draw the deformation
      Image<double> defImg(partImg.getDims(), ZEROS);
      
      float defScale = 500;
      for(int y=0; y<defImg.getHeight(); y++)
        for(int x=0; x<defImg.getWidth(); x++)
        {
          double px = (double)((defImg.getWidth()/2) - x)/20.0;
          double py = (double)((defImg.getHeight()/2) - y)/20.0;

          double val = px*px * part.deformation[0] +
                       px    * part.deformation[1] +
                       py*py * part.deformation[2] +
                       py    * part.deformation[3];
          defImg.setVal(x,y,val*defScale);

        }
      inplacePaste(defImage, toRGB((Image<byte>)defImg), topLeft);
      //Draw a border around the part
      drawRect(defImage, Rectangle(topLeft, partImg.getDims()),
          PixRGB<byte>(255,0,0));

    }
    Layout<PixRGB<byte> > compDisp = hcat(compImage, partsImage);
    compDisp = hcat(compDisp, defImage);

    modelImg = vcat(modelImg, compDisp);
  }

  return modelImg.render();

}

void DPM::readModel(const char* fileName)
{
  FILE *fp = fopen(fileName, "rb");
  if (fp == NULL)
    LFATAL("Can not open model file (%s)", fileName);

  LINFO("Reading model from %s", fileName);

  int numComponents;
  if (fread(&numComponents, sizeof(int), 1, fp) != 1)
    LFATAL("Invalid model file");
  LINFO("Num Components %i", numComponents);

  itsModel = Model();

  for(int c=0; c<numComponents; c++)
  {
    ModelComponent modelComponent;
    
    //Get the root filter
    int filterDims[3];
    if(fread(filterDims, sizeof(int), 3, fp) != 3)
      LFATAL("Invalid model file");
    int width = filterDims[1];
    int height = filterDims[0];
    int numFeatures = filterDims[2];

    ImageSet<double> features;
    for(int feature=0; feature<numFeatures; feature++)
    {
      Image<double> featureMap(width, height, NO_INIT);
      if (fread(featureMap.getArrayPtr(), sizeof(double), width*height, fp) != (uint)(width*height))
        LFATAL("Invalid model file");
      features.push_back(featureMap);
    }

    //get the offset
    double offset = 0;
    if (fread(&offset, sizeof(double), 1, fp) != 1)
      LFATAL("Invalid model file");
    modelComponent.offset = offset;


    modelComponent.rootFilter = features;

    //Get the parts
    int numParts;
    if (fread(&numParts, sizeof(int), 1, fp) != 1)
      LFATAL("Invalid model file");
    LINFO("Reading component %i number of parts %i", c, numParts);
    modelComponent.parts.resize(numParts);
    for(int p=0; p<numParts; p++)
    {
      //Get the anchor
      double anchor[3];
      if(fread(anchor, sizeof(double), 3, fp) != 3)
        LFATAL("Invalid model file");
      modelComponent.parts[p].anchor = Point2D<float>(anchor[0], anchor[1]);
      modelComponent.parts[p].scale = anchor[2];

      //get the deformation
      double deformation[4];
      if(fread(deformation, sizeof(double), 4, fp) != 4)
        LFATAL("Invalid model file");
      modelComponent.parts[p].deformation =
        std::vector<double>(deformation, deformation + 4);

      //Get the features
      int filterDims[3];
      if(fread(filterDims, sizeof(int), 3, fp) != 3)
        LFATAL("Invalid model file");
      int width = filterDims[0];
      int height = filterDims[1];
      int numFeatures = filterDims[2];

      ImageSet<double> features;
      for(int feature=0; feature<numFeatures; feature++)
      {
        Image<double> featureMap(width, height, NO_INIT);
        if (fread(featureMap.getArrayPtr(), sizeof(double), width*height, fp) != (uint)(width*height))
          LFATAL("Invalid model file");
        features.push_back(featureMap);
      }

      modelComponent.parts[p].features = features;
    }

    itsModel.components.push_back(modelComponent);
  }
  fclose(fp);

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
