/*!@file CINNIC/CINNICstatsRun.H Run some stats with CINNIC */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNICstatsRun.H $
// $Id: CINNICstatsRun.H 9412 2008-03-10 23:10:15Z farhan $
//

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#ifndef CINNICSTATSRUN_H_DEFINED
#define CINNICSTATSRUN_H_DEFINED

#include "CINNIC/statsIncl.H"

class CINNICstatsRun
{
protected:
  //! the output file
  std::string statsFile;
  //! used for something
  int ii,jj;
  //! variables that used to be defined, but are now in stats.conf
  int decSize,unEvenDec,lowThresh,highThresh,chopVal;
  //! This is the number of images to be compaired
  int imageCount,zeroCount;
  //! these are used for high quality comparison images
  int qualityImage,qChop;
  //! this is true if the image is to be resized
  int doResize;
  //! the number of salience points to find
  int pointsNum;
  //! width of circles to draw out in sal map
  int circRad;
  //! threshold at which to consider a point
  int rankThresh;
  //! this tells if point and flood should pre-process image #
  int preProcessPNF;
  //! this is set if you want to use draw disk rather then flood. Use w caution
  int useDrawDisk;
  //! the size of the disk to be drawn (r)
  int diskSize;
  //! use this to produce thick mono color circle plots in figure
  int monoColor;
  //! offset for center of image element in output image
  int centerOffset;
  //! how much to attenuate borders for edge effects
  int edgeAtten;
  //! the total error ratio, mean Chamfer34 value
  float totalErrorRatio, meanChamfer;
  //! the mean error ratio, other stats
  float totalMeanError,totalStdError,maxMean,minMean;
  //! The total euclidian distance
  float totEucDist;
  //! the mean euclidian distance, other stats
  float meanEucDist,stdEucDist,maxEuc,minEuc;
  //! total regression
  float totReg;
  //! mean regression, other stats
  float meanReg,stdReg,maxReg,minReg;
  //! flood variabels for image flooding
  float floodThresh, floodVal;
  //! stats values for point and flood
  float foundTotal,foundMean,foundSTD,rankTotal,rankMean,rankSTD;
  //! values for Polat and Sagi
  float PSError;
  //! mu1 and mu2 as the population mean
  float mu1, mu2;
  //! Image to test
  Image<float> testImage;
  //! Image to compare against
  Image<float> compImage;
  //! sal map for PointAndFlood
  Image<float> salMap;
  //! Temp image 1
  Image<byte> tempImage1;
  //! Temp image 2
  Image<byte> tempImage2;
  //! Temp image 1c
  Image<PixRGB <byte> > tempImage1c;
  //! temp image 1cf
  Image<PixRGB <float> > tempImage1cf;
  //! output image of points
  Image<PixRGB <float> > outImageTemplate;
  //! output image of points
  Image<PixRGB <float> > outImageSource;
  //! track the mask image for stats
  Image<float> maskImage;
  //! the count of error pixels in each image
  std::vector<int> errorCount;
  //! the count of total positive pixels in each image
  std::vector<int> totCount;
  //! The total pixel value (/n = average pixel value)
  std::vector<float> totalVal;
  //! The ratio of error pixels to non-error pixels
  std::vector<float> errorRatio;
  //! place holder for vectored images
  std::vector<float> testImageVector;
  //! place holder for vectored images
  std::vector<float> compImageVector;
  //! More stats
  std::vector<float> testMean;
  //! More stats
  std::vector<float> testStd;
  //! More stats
  std::vector<float> compMean;
  //! More stats
  std::vector<float> compStd;
  //! Regression
  std::vector<float> regression;
  //!Eucliduan Distance between images
  std::vector<double> eucDist;
  //! General stats object
  stats<float> Stats;
  //! Another stats object
  stats<double> dStats;
  //! holds max n points from image
  std::vector<Point2D<int> > pointVec;
  //! holds max points for all images
  std::vector< std::vector<Point2D<int> > > maxPointList;
  //! holds max val for all chamfer34 processed images
  std::vector<float> postChamferVal;
  //! holds the count of correct salient points
  std::vector<float> pointsFound;
  //! how many images found this number of salient points
  std::vector<int> pointsFoundNum;
  //! holds the rank of the most salient point on the object
  std::vector<float> pointRank;
  //! how many images are at this rank
  std::vector<int> pointRankNum;
  //Point2D<int> *point2D = new Point2D<int>();
  Point2D<int> *point2D;
  //Point2D<int> *oldPoint;
  //! a lovelly pixRGB instance
  PixRGB<float> pixRGB;
  //! std::vector for number of candidate pixels for point and flood
  std::vector<long> candPixels;
  //! std::vector for the number of real pixels on the contour
  std::vector<long> realPixels;
  //! Vector to hold p for striking a pixel
  std::vector<double> strikeP;
  //! Bernoulli distribution P for the nth strike
  std::vector<double> bernoulliP;
public:
  //! default constructor
  CINNICstatsRun();

  //! default destructor
  ~CINNICstatsRun();

  //! sets the config and loads files
  /*! Read in the configuration file values
    @param config the parsed config file
  */
  void setConfig(readConfig &config, readConfig &config2);

  //! sets a bunch of parameters and junk
  /*! called automatically by constructor, but needs to be called if one
     of the methods such as runStandardStats or runPointAndFlood is called
     more then once.
  */
  void setStuff(readConfig &fileList);

  //! single file option
  void setStuff();

  //! run standard stats battery on images with contour traces
  /*! this method finds how similar an image is to another image based
  upon its corralation, euclidian distance and mean error. It takes in
  two image files and does the compairison. It can run a whole batch of
  images (what its made to do) by setting up the file names in a config
  file
  @param fileList This is the parsed list of files
  */
  void runStandardStats(readConfig &fileList);

  //! run salience point analysis on image data
  /*! This method will attempt to find the most salient points in an
    input image, then it will compair thos points to points that are
    expected. It does this by first finding a point, then finding
    its distance to a contour using champher, then the point are is flooded
    to prevent it from being found again. This is repeated n times
    @param fileList This is the list of input images
  */
  void runPointAndFlood(readConfig &fileList, const char* param);

  //! Run through the input image. Find likelyhood of a random match
  /*! This method will take the compare image and find what is the
    p that a random
    pixel will land on a pixel that is a contour candidate pixel.
    Takes in pointers
    that return p, the number of candidate pixels and the total
    number of pixels
    @param likelyhood This is the p that a random pixel will fall on a candidate contour
    @param posRegionCount This is the number of candidate pixels in the image
    @param totalCount This is the total number of pixels in the image
  */
  void randomMatch(float *likelyhood,long *posRegionCount, long *totalCount);

  //! method for comp image pre-processing
  void preProcess();

  //! checks what to do about images that are of different size
  void checkSize();

  //! this is a table of colors to be used for circles
  PixRGB<float> colorTable(int i);

  //! pointAndFlood on one image, no stats
  void pointAndFloodImage(Image<float> test_image, Image<float> sal_map
                     ,int points,char* filename, float floodv, float floodt);

  //! pointAndFlood on one image, no stats
  void pointAndFloodImage(Image<float> test_image, Image<float> sal_map
                     ,char* filename);

  //! this is the core to point and flood
  void pointAndFlood(const char* filename,int i,bool standalone);

  //! this will take in an image and make a 2AFC, see Itti Thesis 145-149
  /*! This takes in an image and performs a simulation of Polat and Sagi
    1994 two alternative forced choice. It takes in an image and defines
    an error in judgement based upon the 2AFC paradigm. It returns the
    error. This should be compaired to the staircase method described
    by Polat and Sagi in obtaining threshold.
  */
  float polatSagi2AFC(Image<float> targetImage,Image<float> notargetImage);
  //! Return mu1
  float getMu1();
  //! Return mu2
  float getMu2();

};



#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
