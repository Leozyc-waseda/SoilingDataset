/*!@file Features/LocalBinaryPatterns.C Multiclass LocalBinaryPatterns Classifier module */
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
// Primary maintainer for this file: Dan Parks <danielfp@usc.edu>
// $HeadURL$
// $Id$
//

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <map>

#include "LocalBinaryPatterns.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Component/OptionManager.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"

#define LOG_OFFSET 1.0F

namespace
{
inline int lbpsign(int x)
{
  return (((x)<0) ? 0 : 1);
}

};


//! Constructor
LocalBinaryPatterns::LocalBinaryPatterns(int LBPRadius, int LBPPixels, int varBins, bool useColor, bool normalize) :
  itsLBPRadius(LBPRadius),
  itsLBPPixels(LBPPixels),
  itsVarBins(varBins),
  itsUseColor(useColor),
  itsNormalize(normalize)
{
  if(itsUseColor)
    itsColorBins = 2;
  else
    itsColorBins = 0;
}

//! Destructor
LocalBinaryPatterns::~LocalBinaryPatterns()
{
}



//! Add model to

void LocalBinaryPatterns::addModel(const Image<PixRGB<byte> >& texture, const int id)
{
  std::vector<float> lbp,col,var;
  createRawHistogram(texture,lbp,col,var);
  // Bin LBP in a temp list until we are ready to build the complete histogram
  addModel(lbp,col,var,id);
}
 
void LocalBinaryPatterns::addModel(const std::vector<float>& lbp, const std::vector<float> &col, const std::vector<float>& var, const int id)
{
  // Store LBP and color bins 
  itsTempLBP[id].push_back(lbp);
  itsTempColor[id].push_back(col);
  // Sort variance, to speed up build process
  std::vector<float> svar = var;
  std::sort(svar.begin(),svar.end());
  // Store Variance values temporarily until we can get the total distribution of variances
  itsTempVariance[id].push_back(svar);
  //LINFO("Number of models is now %Zu, for id %d, num exemplars %Zu ",itsTempLBP.size(),id,itsTempLBP[id].size());
}

std::vector<float> LocalBinaryPatterns::createHistogram(const Image<PixRGB<byte> >& texture)
{
  std::vector<float> lbp,col,var;
  createRawHistogram(texture,lbp,col,var);
  std::vector<float> varHist = convertVariance(var);
  std::vector<float> hist = lbp;
  hist.insert(hist.end(),col.begin(),col.end());
  hist.insert(hist.end(),varHist.begin(),varHist.end());
  return hist;
}


void LocalBinaryPatterns::createRawHistogram(const Image<PixRGB<byte> >& colTexture, std::vector<float>& lbps, std::vector<float>& col, std::vector<float>& vars)
{
  Image<float> texture = luminance(colTexture);
  int w = texture.getWidth(), h = texture.getHeight();
  lbps.resize(itsLBPPixels+2);
  float step = 1;//itsLBPPixels*2+1;
  ASSERT(2*itsLBPRadius < std::min(w,h));
  float *g = new float[itsLBPPixels];
  int *sg = new int[itsLBPPixels];
  int sumEntries=0;
  for (float y = itsLBPRadius; y < h-itsLBPRadius; y+=step)
    {
      for (float x = itsLBPRadius; x < w-itsLBPRadius; x+=step)
	{
	  float var=0;
	  //float lbp=0;
	  uint diff=0;
	  float uniform=0;
	  float mu=0;
	  float gc = texture.getValInterp(x,y);
	  for(int p=0;p<itsLBPPixels;p++)
	    {
	      g[p] = texture.getValInterp(x-itsLBPRadius*sin(2.0*M_PI*p/itsLBPPixels),y+itsLBPRadius*cos(2.0*M_PI*p/itsLBPPixels));
	      mu += g[p];
	      sg[p] = lbpsign(g[p]-gc);
	      // If want to calculate full LBP, need to do
	      //lbp += lbpsign(g[p])*pow(2,p);
	      // then circular bit shift until minimum value is found
	      // Here we will just bin the "uniform" patterns in addition to the variance
	      diff += sg[p];	      
	    }
	  for(int p=0;p<itsLBPPixels;p++)
	    {
	      if(p==0) uniform += abs(sg[itsLBPPixels-1]-sg[0]);
	      else uniform += abs(sg[p]-sg[p-1]);
	      var += (sg[p]-mu)*(sg[p]-mu);
	    }
	  // Bin into one of the uniform bins or into the "other" bin
	  sumEntries++;
	  if(uniform<=2) lbps[diff]++;
	  else lbps[itsLBPPixels+1]++;
	  // Normalize and add to list of variances (have to wait to bin variances until a total distribution is determined)
	  if(itsVarBins > 0)
	    vars.push_back(var / itsLBPPixels);
	}
    }
  // Normalize values in lbps histogram
  if(itsNormalize)
    {
      for(uint b=0;b<lbps.size();b++)
	{
	  lbps[b]=lbps[b]/float(sumEntries)+LOG_OFFSET;
	}
    }
  delete[] g;
  delete[] sg;

  // If using color, add color

  if(itsUseColor)
    {
      float rg,by,rgbysum;
      colorSum(colTexture,colTexture.getBounds(),rg,by);
      if(itsNormalize)
	{
	  rgbysum = rg+by;
	  col.push_back(rg/rgbysum+LOG_OFFSET);
	  col.push_back(by/rgbysum+LOG_OFFSET);
	}
      else
	{
	  col.push_back(rg);
	  col.push_back(by);
	}
    }
}


void LocalBinaryPatterns::getLabeledData(const MapModelVector& models, std::vector<std::vector<float> >& data, std::vector<float>& labels)
{
  data.clear();
  labels.clear();
  MapModelVector::const_iterator mitr=models.begin(), mstop = models.end();  
  int idx=0;
  for(;mitr!=mstop;mitr++)
    {
      // Iterate through model exemplars
      for(uint i=0;i<mitr->second.size();i++)
	{
	  labels.push_back(mitr->first);
	  data.push_back(std::vector<float>());
	  data[idx] = mitr->second[i];
	  idx++;
	}
    }

}

void LocalBinaryPatterns::colorSum(Image<PixRGB<byte> >img, Rectangle rec, float& rg, float& by)
{
  ASSERT(img.rectangleOk(rec));
  const Image<PixRGB<byte> > subImg = crop(img,rec);
  Image<float> l,a,b;
  getLAB(subImg,l,a,b);
  rg=0; by=0;
  Dims dims = subImg.getDims();

  for (int y = 0; y < dims.h(); ++y)
    {
      for (int x = 0; x < dims.w(); ++x)
	{
	  rg+=(a.getVal(x,y)+128);
	  by+=(b.getVal(x,y)+128);
	}
    }

}



std::vector<float> LocalBinaryPatterns::convertVariance(const std::vector<float>& vars)
{
  std::vector<float> varHist = std::vector<float>(vars.size());
  // Convert the variance values into a histogram based on the bin break points
  varHist.resize(itsVarBins);
  if(itsVarBins == 0)
    return varHist;
  int sumEntries=vars.size();
  int binHint = -1;
  for(uint i=0;i<vars.size();i++)
    {
      // Optimized version
      int bin = getVarIndex<std::vector<float> >(itsVarBinThresholds,vars[i],binHint);
      varHist[bin]++;
      binHint=bin;
    }
  // Normalize values in var histogram
  for(uint b=0;b<varHist.size();b++)
    {
      varHist[b]=varHist[b]/float(sumEntries)+LOG_OFFSET;
    }
  return varHist;
}

LocalBinaryPatterns::MapModelVector LocalBinaryPatterns::getModels()
{
  return itsModels;
}

void LocalBinaryPatterns::setModels(const MapModelVector& models)
{
  itsModels=models;
}

std::vector<float> LocalBinaryPatterns::getVarThresholds()
{
  return itsVarBinThresholds;
}

void LocalBinaryPatterns::setVarThresholds(std::vector<float> thresholds)
{
  ASSERT(thresholds.size()==(uint)std::max(0,itsVarBins-1));
  itsVarBinThresholds = thresholds;
}


void LocalBinaryPatterns::setIncompleteModels(const MapModelVector& incompleteModels)
{
  // Need to split the vectors into an LBP operator and a variance list based on the size of the LBP operator histogram
  MapModelVector::const_iterator imitr=incompleteModels.begin(), imstop = incompleteModels.end();
  // Iterate through model ids
  for(;imitr!=imstop;imitr++)
    {
      // Iterate through model exemplars
      for(uint i=0;i<imitr->second.size();i++)
	{
	  std::vector<float> combinedHist = imitr->second[i];
	  // Split according to length of LBP histogram
	  std::vector<float> lbp,col,tmpVar;
	  lbp.insert(lbp.begin(),combinedHist.begin(),combinedHist.begin()+itsLBPPixels+2);
	  col.insert(col.begin(),combinedHist.begin()+itsLBPPixels+2,combinedHist.begin()+itsLBPPixels+2+itsColorBins);
	  tmpVar.insert(tmpVar.begin(),combinedHist.begin()+itsLBPPixels+2+itsColorBins,combinedHist.end());
	  addModel(lbp,col,tmpVar,imitr->first);
	}
    }
}

LocalBinaryPatterns::MapModelVector LocalBinaryPatterns::getIncompleteModels()
{
  MapModelVector models;
  MapModelVector::const_iterator imitr=itsTempLBP.begin(), imstop = itsTempLBP.end();
  // Iterate through model ids
  for(;imitr!=imstop;imitr++)
    {
      // Iterate through model exemplars
      for(uint i=0;i<imitr->second.size();i++)
	{
	  std::vector<float> lbp = imitr->second[i];
	  std::vector<float> col = itsTempColor[imitr->first][i];
	  std::vector<float> tmpVar = itsTempVariance[imitr->first][i];
	  // Combined lbp, col, and var
	  std::vector<float> combinedHist = lbp;
	  combinedHist.insert(combinedHist.end(),col.begin(),col.end());
	  combinedHist.insert(combinedHist.end(),tmpVar.begin(),tmpVar.end());
	  models[imitr->first].push_back(combinedHist);
	}
    }
  return models;
}

void LocalBinaryPatterns::combineModels(const std::vector< MapModelVector >& allModels, MapModelVector& combined)
{
  for(uint i=0;i<allModels.size();i++)
    {
      appendMap(combined,allModels[i]);
    }
}

void LocalBinaryPatterns::appendMap(MapModelVector& dst, const MapModelVector& src)
{
  MapModelVector::const_iterator sitr=src.begin(), sstop = src.end();
  // Iterate through model ids
  for(;sitr!=sstop;sitr++)
    {
      // Iterate through model exemplars
      for(uint i=0;i<sitr->second.size();i++)
	{
	  ASSERT(dst[sitr->first].size() <= sitr->second.size());
	  // Allocate space if needed
	  if(dst[sitr->first].size() < sitr->second.size())
	    dst[sitr->first].resize(sitr->second.size());
	  // Append map exemplar to end of original exemplar
	  dst[sitr->first][i].insert(dst[sitr->first][i].end(),sitr->second[i].begin(),sitr->second[i].end());
	}
    }
}

std::vector<float> LocalBinaryPatterns::merge(const std::vector<float>& left, const std::vector<float>& right)
{
    // Fill the resultant vector with sorted results from both vectors
    std::vector<float> result;
    unsigned left_it = 0, right_it = 0;
 
    while(left_it < left.size() && right_it < right.size())
    {
        // If the left value is smaller than the right it goes next
        // into the resultant vector
        if(left[left_it] < right[right_it])
        {
            result.push_back(left[left_it]);
            left_it++;
        }
        else
        {
            result.push_back(right[right_it]);
            right_it++;
        }
    }
 
    // Push the remaining data from both vectors onto the resultant
    while(left_it < left.size())
    {
        result.push_back(left[left_it]);
        left_it++;
    }
 
    while(right_it < right.size())
    {
        result.push_back(right[right_it]);
        right_it++;
    }
 
    return result;
}

void LocalBinaryPatterns::buildModels()
{
  itsVarBinThresholds.clear();
  if(itsVarBins > 0)
    {
      // Build total variance distribution, note this assumes that individual variance vectors were presorted
      std::vector<float> totalDist;
      MapModelVector::const_iterator tvitr=itsTempVariance.begin(), tvstop = itsTempVariance.end();
      for(;tvitr!=tvstop;tvitr++)
	{
	  for(uint i=0;i<tvitr->second.size();i++)
	    {
	      totalDist = merge(totalDist,tvitr->second[i]);
	    }
	}
      // Determine number of elements to skip to collect thresholds
      float step = totalDist.size()/float(itsVarBins);
      // Grab thresholds (might want to interpolate them)
      for(int s=1;s<itsVarBins;s++)
	{
	  float thresh=totalDist[floor(step*s)];
	  itsVarBinThresholds.push_back(thresh);
	}
    }
  itsModels.clear();
  convertIncompleteModels();
}

void LocalBinaryPatterns::convertIncompleteModels()
{
  MapModelVector varHist; 
  MapModelVector::const_iterator tvitr=itsTempVariance.begin(), tvstop = itsTempVariance.end();
  if(itsVarBins > 0)
    {
      // Convert model variance data to histograms
      for(tvitr=itsTempVariance.begin();tvitr!=tvstop;tvitr++)   
	{
	  std::vector<std::vector<float> > bins = std::vector<std::vector<float> > ((*tvitr).second.size());
	  for(uint i=0;i<tvitr->second.size();i++)
	    {
	      bins[i] = convertVariance(tvitr->second[i]);
	    }
	  varHist[tvitr->first] = bins;
	}
    }
  
  MapModelVector::const_iterator tmitr=itsTempLBP.begin(), tmstop = itsTempLBP.end();
  for(;tmitr!=tmstop;tmitr++)
    {
      for(uint i=0;i<tmitr->second.size();i++)
	{
	  std::vector<float> hist = tmitr->second[i];
	  std::vector<float> col = itsTempColor[tmitr->first][i];
	  // Add color
	  hist.insert(hist.end(),col.begin(),col.end());
	  // Add variance if valid
	  if(itsVarBins > 0)
	    hist.insert(hist.end(),varHist[tmitr->first][i].begin(),varHist[tmitr->first][i].end());
	  // Initialize if no key is present
	  if(itsModels.find(tmitr->first) == itsModels.end())
	    itsModels[tmitr->first] = std::vector<std::vector<float> >();
	  itsModels[tmitr->first].push_back(hist);
	}
    }

}


template<class T> int LocalBinaryPatterns::getVarIndex(const T& thresh, float var, int binHint)
{
  // Optimization, if variance is pre-sorted
  if(binHint >=0)
    {
      if(binHint < int(thresh.size()) )
	{
	  if(var < thresh[binHint])
	    {
	      if(binHint == 0)
		return binHint;
	      else if(var > thresh[binHint-1])
		return binHint;
	    }
	}
      else if(var > thresh[binHint-1])
	{
	  return binHint;
	}
    }
  // Binary search to find correct bin
  bool binFound=false;
  const int endBin = thresh.size();
  // Check if only one bin
  if(endBin==0)
    return 0;
  int lowBin = 0, highBin = endBin;
  int curBin = (lowBin+highBin)/2;
  while(!binFound)
    {
      
      if(var > thresh[curBin])
	lowBin = curBin+1;
      else 
	highBin = curBin;
      curBin = (lowBin+highBin)/2;
      if( highBin - lowBin <= 1)
	{
	  binFound=true;
	}
    }
  if(var > thresh[lowBin])
    curBin = highBin;
  else
    curBin = std::min(lowBin,endBin);
  return curBin;
}

//! Get number of models
uint LocalBinaryPatterns::getTotalModelExemplars(MapModelVector models)
{
  MapModelVector::const_iterator mmitr = models.begin(), mmstop = models.end();
  uint cnt=0;
  for(;mmitr!=mmstop;mmitr++)
    {
      cnt += mmitr->second.size();
    }
  return cnt;
}


LocalBinaryPatterns::MapModelVector LocalBinaryPatterns::readModelsFile(std::string modelsFile)
{
  FILE *fmodel;
  int numEntries, numColumns;
  MapModelVector models;

  fmodel = fopen(modelsFile.c_str(),"r");
  if(fmodel == NULL)
    {
      LFATAL("Unable to open LBP models file");
    }
  numEntries=0;
  while(1)
    {
      int id;
      if(fscanf(fmodel, "%d %d ", &id,&numColumns)!= 2) 
	{
	  LINFO("Read %d samples in model file %s",numEntries,modelsFile.c_str());
	  break;
	}
      std::vector<float> hist;
      for(int c=0;c<numColumns;c++)
	{
	  float tmp;
	  if(fscanf(fmodel, "%f ", &tmp)!= 1) LFATAL("Failed to load column %d in row %d of file: %s",c,numEntries,modelsFile.c_str());
	  hist.push_back(tmp);
	}
      if(fscanf(fmodel,"\n")!= 0) LFATAL("Failed to parse newline at end of row %d of file: %s",numEntries,modelsFile.c_str());
      // Initialize if no key is present
      if(models.find(id) == models.end())
	models[id] = std::vector<std::vector<float> >();
      models[id].push_back(hist);
      numEntries++;
    }
  fclose(fmodel);
  return models;
}

void LocalBinaryPatterns::writeModelsFile(std::string modelsFile, MapModelVector models)
{
  std::ofstream outfile;
  outfile.open(modelsFile.c_str(),std::ios::out);
  if (outfile.is_open()) 
    {
      MapModelVector::const_iterator mmitr = models.begin(), mmstop = models.end();
      for(;mmitr!=mmstop;mmitr++)    
	{
	  for(uint i=0;i<mmitr->second.size();i++) 
	    {
	      outfile << mmitr->first << " " << mmitr->second[i].size() << " ";
	      for(uint j=0;j<mmitr->second[i].size();j++) 
		{
		  outfile << std::setiosflags(std::ios::fixed) << std::setprecision(4) << mmitr->second[i][j] << " ";
		}
	      outfile << std::endl;
	    }
	}
      outfile.close();
    }
  else 
    {
      LFATAL("Could not open LBP Models output file");
    }
}


std::vector<float> LocalBinaryPatterns::readThresholdsFile(std::string thresholdsFile)
{
  FILE *fthresh;
  int numEntries;
  std::vector<float> thresholds;

  fthresh = fopen(thresholdsFile.c_str(),"r");
  if(fthresh == NULL)
    {
      LFATAL("Unable to open Bin Thresholds input file");
    }

  if(fscanf(fthresh, "%d\n", &numEntries) != 1) LFATAL("Failed to load number of entries from: %s", thresholdsFile.c_str());
  for(int i=0;i<numEntries;i++)
    {
      float tmp;
      if(fscanf(fthresh, "%f ", &tmp)!= 1) LFATAL("Failed to load threshold for bin %d in file: %s",i,thresholdsFile.c_str());
      thresholds.push_back(tmp);
    }
  if(fscanf(fthresh,"\n")!= 0) LFATAL("Failed to parse newline at end of bin thresholds in file: %s",thresholdsFile.c_str());
  fclose(fthresh);
  return thresholds;
}

void LocalBinaryPatterns::writeThresholdsFile(std::string thresholdsFile, std::vector<float> thresholds)
{
  std::ofstream outfile;
  outfile.open(thresholdsFile.c_str(),std::ios::out);
  if (outfile.is_open()) 
    {
      outfile << thresholds.size() << " ";
      outfile << std::endl;
      for(uint i=0;i<thresholds.size();i++) 
	{
	  outfile << std::setiosflags(std::ios::fixed) << std::setprecision(4) << thresholds[i] << " ";
	}
      outfile << std::endl;
      outfile.close();
    }
  else 
    {
      LFATAL("Could not open Bin Thresholds output file");
    }
}


// Template instantiations for getVarIndex
template int LocalBinaryPatterns::getVarIndex(const std::vector<float> &thresh, float var, int binHint);

template int LocalBinaryPatterns::getVarIndex(const std::deque<float> &thresh, float var, int binHint);
