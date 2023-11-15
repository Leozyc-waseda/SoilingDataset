/*!@file plugins/SceneUnderstanding/V2.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V2.C $
// $Id: V2.C 14683 2011-04-05 01:30:59Z lior $
//

#ifndef V2_C_DEFINED
#define V2_C_DEFINED

#include "plugins/SceneUnderstanding/V2.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/Convolutions.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueue.H"
#include "Util/CpuTimer.H"
#include "Util/JobServer.H"
#include "Util/JobWithSemaphore.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <stdio.h>

const ModelOptionCateg MOC_V2 = {
  MOC_SORTPRI_3,   "V2-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_V2ShowDebug =
  { MODOPT_ARG(bool), "V2ShowDebug", &MOC_V2, OPTEXP_CORE,
    "Show debug img",
    "v2-debug", '\0', "<true|false>", "false" };

const ModelOptionDef OPT_V2TrainNFA =
  { MODOPT_ARG(bool), "V2TrainNFA", &MOC_V2, OPTEXP_CORE,
    "Train the Number Of False Alarms (NFA)",
    "v2-trainNFA", '\0', "<true|false>", "false" };

const ModelOptionDef OPT_V2TrainNFAFile =
  { MODOPT_ARG(std::string), "V2TrainNFAFile", &MOC_V2, OPTEXP_CORE,
    "Train the Number Of False Alarms (NFA) output file",
    "v2-trainNFAFile", '\0', "<string>", "NFATrain.dat" };

//Define the inst function name
SIMMODULEINSTFUNC(V2);

namespace
{

  typedef std::map<size_t,size_t> BinPDFMap;

  void writePDF(std::vector< std::vector<NFAInfo> > pdf)
  {
    LINFO("Dump Data");
    FILE *fp = fopen("BinPDFMap.dat", "w");
    for(uint i=0; i<pdf.size(); i++)
    {
      for(uint j=0; j<pdf[i].size(); j++)
      {
        fprintf(fp, "%i %i %i %i %i %i %i\n",  i,
            pdf[i][j].ori,
            pdf[i][j].xLoc,
            pdf[i][j].yLoc,
            pdf[i][j].length,
            pdf[i][j].pts,
            pdf[i][j].alg);
      }
    }
    fclose(fp);
  }

  void writePDF(std::vector<NFAPDFMap>& pdf,const char* filename)
  {
    LINFO("Dump Data");
    FILE *fp = fopen(filename, "w");
    for(uint i=0; i<pdf.size(); i++)
    {
      NFAPDFMap::iterator itr1;
      for(itr1=pdf[i].begin(); itr1 != pdf[i].end(); itr1++)
      {
        std::map<short int, //xLoc
          std::map<short int, //yLoc,
          std::map<short int, //length, 
          std::map<short int, //alg, 
          NumEntries > > > >::iterator itr2;
        for(itr2=itr1->second.begin(); itr2 != itr1->second.end(); itr2++)
        {
          std::map<short int, //yLoc,
            std::map<short int, //length, 
            std::map<short int, //alg, 
            NumEntries > > >::iterator itr3;
          for(itr3=itr2->second.begin(); itr3 != itr2->second.end(); itr3++)
          {
            std::map<short int, //length, 
              std::map<short int, //alg, 
              NumEntries > >::iterator itr4;
            for(itr4=itr3->second.begin(); itr4 != itr3->second.end(); itr4++)
            {
              std::map<short int, //alg, 
                NumEntries >::iterator itr5;
              for(itr5=itr4->second.begin(); itr5 != itr4->second.end(); itr5++)
              {
                fprintf(fp, "%i %i %i %i %i %i %i %zu\n", 
                    i, 
                    itr1->first, itr2->first, itr3->first, itr4->first, itr5->first,
                    itr5->second.pts, itr5->second.num);
              }

            }
          }

        }

      }
    }
    fclose(fp);
  }

  class TrainNFAJob : public JobWithSemaphore
  {
    public:

      TrainNFAJob(V2* v2,Image<float> angles,
          const std::vector<Point2D<int> >& pos,
          int rectWidth, int startWidth, 
          NFAPDFMap& pdf) :
        itsV2(v2),
        itsAngles(angles),
        itsPositions(pos),
        itsRectWidth(rectWidth),
        itsStartWidth(startWidth),
        itsPDF(pdf)
    {}

      virtual ~TrainNFAJob() {}

      virtual void run()
      {
        int width = itsAngles.getWidth();
        int height = itsAngles.getHeight();
        int maxLength = sqrt(width*width + height*height);
        LINFO("Start thread for %i", itsRectWidth);

        for(int ori  = -180; ori < 180; ori += 4)
        {
          for(uint locIdx=0; locIdx<(itsPositions.size()/250); locIdx++)
          {
            for(int length=4; length<maxLength; length+=4)
            {
              Dims size(length, itsRectWidth);
              Rect rect(itsPositions[locIdx], size, (float)ori*M_PI/180.0);

              int pts=0;
              int alg=0;
              int x1=0,x2=0,y1=0;

              for(rect.initIter(); !rect.iterEnd(); rect.incIter(x1,x2,y1))
              {
                for(int x = x1; x <= x2; x++)
                {
                  ++pts;
                  if( x>=0 && y1>=0 &&
                      x<width && y1<height )
                  {
                    if( itsV2->isaligned(Point2D<int>(x,y1),
                          itsAngles,ori,M_PI/20) )
                      ++alg;
                  }
                }
              }

              if (alg > 0)
              {
                const short int xPos = itsPositions[locIdx].i;
                const short int yPos = itsPositions[locIdx].j;
                NumEntries& info = itsPDF[ori][xPos][yPos][length][alg];
                info.num++;
                info.pts = pts;
              }
            }
          }
          LINFO("Ori %i", ori);
        }
        LINFO("Done %i: size %zu", itsRectWidth, itsPDF.size());
        this->markFinished();
      }

      virtual const char* jobType() const { return "TrainNFAJob"; }
      V2* itsV2;
      Image<float> itsAngles;
      const std::vector<Point2D<int> >& itsPositions;
      int itsRectWidth;
      int itsStartWidth;
      NFAPDFMap& itsPDF; 
  };
}



// ######################################################################
V2::V2(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV1Output),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_V2ShowDebug, this),
  itsTrainNFA(&OPT_V2TrainNFA, this),
  itsTrainNFAFile(&OPT_V2TrainNFAFile, this),
  itsLSDVerbose(false),
  itsQuantError(2.0),
  itsAngleTolerance(20.0),
  itsEPS(0)
  //itsStorage(cvCreateMemStorage(0))
{
  //itsThreadServer.reset(new WorkThreadServer("V2", 20));
  itsPDF.resize(20);
 // itsStoredFrames = "true";
 
  //Pick random locations to sample
  int width = 640;
  int height = 480;
  for(int y=0; y<height; y++)
    for(int x=0; x<width; x++)
      itsLocations.push_back(Point2D<int>(x,y));
  randShuffle(&itsLocations[0], itsLocations.size());


}

// ######################################################################
V2::~V2()
{
  //cvReleaseMemStorage(&itsStorage);
}

// ######################################################################
void V2::init(Dims numCells)
{

}

// ######################################################################
void V2::onSimEventV1Output(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV1Output>& e)
{
  itsV1EdgesState = e->getEdgesState();

  if (SeC<SimEventLGNOutput> lgn = q.check<SimEventLGNOutput>(this))
    itsLGNInput = lgn->getCells();
  
  evolve(q);


}

// ######################################################################
void V2::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "V2", FrameInfo("V2", SRC_POS));
    }
}

// ######################################################################
void V2::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (strcmp(e->getWinName(), "V2"))
    return;

  switch(e->getKey())
  {
    case 10: //1
      itsQuantError += 1;
      break;
    case 24: //q
      itsQuantError -= 1;
      break;
    case 11: //1
      itsAngleTolerance += 1;
      break;
    case 25: //q
      itsAngleTolerance -= 1;
      break;
    case 12: //1
      itsEPS += 1;
      break;
    case 26: //q
      itsEPS -= 1;
      break;
    default:
      break;
  }

  LINFO("Q %f A %f EPS %f",
      itsQuantError, itsAngleTolerance,
      itsEPS);

  //itsQuantError(2.0),
  //itsAngleTolerance(20.0),
  //itsEPS(0),
  
  
  if (e->getMouseClick().isValid())
  {
  }

  evolve(q);

}

void V2::evolve(SimEventQueue& q)
{
  evolveLines();

  //Find corners and symetries
  //cornerSymetryDetection(itsLines);
  //evolveContours();
  //evolveBorderOwner();
  
  q.post(rutz::make_shared(new SimEventV2Output(this, itsLines,
          itsCornersState, itsCornersProposal, itsEdges, itsTensorFields, itsEdges.getDims())));
}

void V2::evolveBorderOwner()
{
  std::vector<ContourState> contours = proposeContoursBO(itsLines);
  itsContours = contours;
}

std::vector<V2::ContourState> V2::proposeContoursBO(std::vector<LineSegment>& lines)
{

  std::vector<ContourState> contours;

  //Propose countors based on border ownership


  //Group all lines that have the same color together
  std::list<LineSegment> currentLines;
  for(uint i=0; i<lines.size(); i++)
    currentLines.push_back(lines[i]);

  //Pick a line and find all matches

  while(!currentLines.empty())
  {
    ContourState contour;
    LineSegment ls = currentLines.front();
    currentLines.pop_front();
    contour.lines.push_back(ls);

    std::list<LineSegment>::iterator iter;
    for(iter = currentLines.begin(); iter != currentLines.end(); )
    {
      if (ls.colorDist(*iter) < 50.0F)
      {
        contour.lines.push_back(*iter);
        currentLines.erase(iter++); //increment the iter after erasing
      } else {
        iter++;
      }
    }

    contours.push_back(contour);
  }

  return contours;
}

void V2::evolveContours()
{

  //Link lines

  itsContours.clear();

  std::vector<ContourState> contours = proposeContours(itsLines);

  //Take the input as the perception for now
  itsContours = contours;
}

std::vector<V2::ContourState> V2::proposeContours(std::vector<LineSegment>& lines)
{

  std::vector<ContourState> contours;

  std::list<LineSegment> currentLines;

  for(uint i=0; i<lines.size(); i++)
    currentLines.push_back(lines[i]);

  while(!currentLines.empty())
  {
    LineSegment ls = currentLines.front();
    currentLines.pop_front();

    Point2D<float> p1 = ls.p1;
    Point2D<float> p2 = ls.p2;

    LINFO("End Points %0.2fx%0.2f %0.2fx%0.2f",
        p1.i, p1.j,
        p2.i, p2.j);

    ContourState contour;
    contour.lines.push_back(ls);

    std::stack<Point2D<float> > endPoints;
    endPoints.push(p1);
    endPoints.push(p2);

    Image<PixRGB<byte> > contoursImg = itsLGNInput[0];
    drawLine(contoursImg, (Point2D<int>)p1, (Point2D<int>)p2,
        PixRGB<byte>(255,0,0), 1);

    //SHOWIMG(contoursImg);

    //std::vector<LineSegment> nerestLines = findNerestLines(currentLines, endPoints);
    //LINFO("Found %lu closest lines", nerestLines.size());

    //while(nerestLines.size() > 0)
    //{
    //  for(uint i=0; i<nerestLines.size(); i++)
    //  {
    //    contour.lines.push_back(nerestLines[i]);

    //    LineSegment& ls = nerestLines[i];
    //    drawLine(contoursImg, (Point2D<int>)ls.p1,
    //        (Point2D<int>)ls.p2, PixRGB<byte>(0,255,0), 1);
    //  }
    //  SHOWIMG(contoursImg);
    //  nerestLines = findNerestLines(currentLines, endPoints);
    //  LINFO("Found Again %lu closest lines", nerestLines.size());

    //}

    //if (contour.lines.size() > 1)
    //  contours.push_back(contour);

    LINFO("Loop\n");
  }


  return contours;

}

std::vector<V2::LineSegment> V2::findNerestLines(std::list<LineSegment>& lines,
    std::stack<Point2D<float> >& endPoints)
{

  std::vector<LineSegment> nerestLines;

  Point2D<float> p = endPoints.top();
  endPoints.pop();

  //Find the closest line to the endpoint
  //float thresh = 5.0;
  //std::list<LineSegment>::iterator iter;

  //for(iter = lines.begin(); iter != lines.end(); iter++)
  //{
  //  if (p.distance(iter->p1) < thresh)
  //  {
  //    endPoints.push(iter->p2);
  //    nerestLines.push_back(*iter);
  //    lines.erase(iter);
  //  } else if (p.distance(iter->p2) < thresh)
  //  {
  //    nerestLines.push_back(*iter);
  //    endPoints.push(iter->p1);
  //    lines.erase(iter);
  //  }
  //}

  return nerestLines;

}

void V2::evolveLines()
{


  std::vector<LineSegment> lines;

  if (itsTrainNFA.getVal())
  {
    double logNT = 0;
    double p = 1.0/20.0;
    //print out the raguler NFA computations
    for(uint n=1; n<18000; n++)
    {
      for(uint k=1; k<n && k<3000; k++)
      {
        double nfaValue = nfa(n, k, p, logNT);
        if (!std::isinf(nfaValue) || nfaValue != 0)
          printf("%i %i %f\n", k, n, nfaValue);

      }
    }
    //trainNFA(itsV1EdgesState);

  } else {
    //Use the LGN Input
    //std::vector<LineSegment> lines = proposeLineSegments(itsLGNInput);

    //Use the V1 Input
    lines = proposeLineSegments(itsV1EdgesState);

  }


  ////For each proposed line find the most Likely line
  ////segment in our current perception
  //for(uint i=0; i<lines.size(); i++)
  //{
  //  float prob;
  //  int lineIdx = findMostProbableLine(lines[i], prob);

  //  if (lineIdx >= 0)
  //  {
  //    //Line Found, update it
  //    itsLines[lineIdx] = lines[i];


  //    //Update the strength
  //    itsLines[lineIdx].strength += 0.5;
  //    if ( itsLines[lineIdx].strength > 1)
  //      itsLines[lineIdx].strength = 1;
  //  } else {
  //    //Add the New line to the current perception
  //    lines[i].strength = 0.5;

  //    itsLines.push_back(lines[i]);
  //  }
  //}

  ////Leaky itegrator line
  ////Slowly kill off any lines that are in the current perception but not in the data
  //for(uint i =0; i<itsLines.size(); i++)
  //{
  //  itsLines[i].strength -= 0.10;

  //  if (itsLines[i].strength < 0)
  //  {
  //    //Remove the line for the perception
  //  }
  //}


  itsLines = lines;

}


int V2::findMostProbableLine(const LineSegment& line, float& prob)
{

  for(uint i=0; i<itsLines.size(); i++)
  {
    if (itsLines[i].distance(line) < 5)
    {
      prob = itsLines[i].distance(line);
      return i;
    }
  }

  return -1;
}

std::vector<V2::LineSegment> V2::proposeLineSegments(ImageSet<float> &LGNInput)
{

  std::vector<LineSegment> lines;

  LINFO("Evolve tensors");
  std::vector<LineSegment> linesLum = lineSegmentDetection(LGNInput[0]);
  std::vector<LineSegment> linesRG = lineSegmentDetection(LGNInput[1]);
  std::vector<LineSegment> linesBY = lineSegmentDetection(LGNInput[2]);
  LINFO("Done");

  for(uint i=0; i<linesLum.size(); i++)
  {
    linesLum[i].color=0;
    lines.push_back(linesLum[i]);
  }
  for(uint i=0; i<linesRG.size(); i++)
  {
    linesRG[i].color=1;
    lines.push_back(linesRG[i]);
  }
  for(uint i=0; i<linesBY.size(); i++)
  {
    linesBY[i].color=2;
    lines.push_back(linesBY[i]);
  }


  return lines;

}

std::vector<V2::LineSegment> V2::proposeLineSegments(V1::EdgesState& edgesState)
{

  //Preform TensorVoting

  //Preform nonMax surpression if true
  TensorField lumTensorField = itsLumTV.evolve(edgesState.lumTensorField, false);
  TensorField rgTensorField  = itsRGTV.evolve(edgesState.rgTensorField, false);
  TensorField byTensorField  = itsBYTV.evolve(edgesState.byTensorField, false);

  itsTensorFields.clear();
  itsTensorFields.push_back(itsLumTV);
  itsTensorFields.push_back(itsRGTV);
  itsTensorFields.push_back(itsBYTV);


  EigenSpace eigen = getTensorEigen(lumTensorField);
  itsEdges = eigen.l1-eigen.l2;
  itsCornersProposal = eigen.l2;

  //Get the max edge from all color spaces
  itsMaxTF = lumTensorField;
  //itsMaxTF.setMax(rgTensorField);
  //itsMaxTF.setMax(byTensorField);


  std::vector<LineSegment> lines = lineSegmentDetection(itsMaxTF, itsQuantError, itsAngleTolerance, itsEPS);
  //std::vector<LineSegment> linesRG = lineSegmentDetection(rgTensorField, itsQuantError, itsAngleTolerance, itsEPS);
  //std::vector<LineSegment> linesBY = lineSegmentDetection(byTensorField, itsQuantError, itsAngleTolerance, itsEPS);


  //for(uint i=0; i<linesLum.size(); i++)
  //{
  //  linesLum[i].color=0;
  //  lines.push_back(linesLum[i]);
  //}
  //for(uint i=0; i<linesRG.size(); i++)
  //{
  //  linesRG[i].color=0;
  //  lines.push_back(linesRG[i]);
  //}
  //for(uint i=0; i<linesBY.size(); i++)
  //{
  //  linesBY[i].color=0;
  //  lines.push_back(linesBY[i]);
  //}


  return lines;

}


V2::rect V2::getRect(const Point2D<int> p1, const Point2D<int> p2, int width)
{

  rect rec;
  // set the first and second point of the line segment
  if (p1.j > p2.j)
  {
    rec.x1 = p2.i*2; rec.y1 = p2.j*2;
    rec.x2 = p1.i*2; rec.y2 = p1.j*2;
  } else {
    rec.x1 = p1.i*2; rec.y1 = p1.j*2;
    rec.x2 = p2.i*2; rec.y2 = p2.j*2;
  }

  double theta = atan2(p2.j-p1.j,p2.i-p1.i);
  if (theta<0)
    theta += M_PI;

  rec.theta = theta;
  rec.dx = (float) cos( (double) rec.theta );
  rec.dy = (float) sin( (double) rec.theta );
  rec.width=width;
  rec.prec = M_PI / 20; /* tolerance angle */
  rec.p = 1.0 / (double) 20;
  return rec;
}

void V2::trainNFA(V1::EdgesState& edgesState)
{

  //Preform TensorVoting

  //Preform nonMax surpression if true
  TensorField lumTensorField = itsLumTV.evolve(edgesState.lumTensorField, false);
  //TensorField rgTensorField  = itsRGTV.evolve(edgesState.rgTensorField, false);
  //TensorField byTensorField  = itsBYTV.evolve(edgesState.byTensorField, false);

  itsTensorFields.clear();
  itsTensorFields.push_back(itsLumTV);
  //itsTensorFields.push_back(itsRGTV);
  //itsTensorFields.push_back(itsBYTV);

  //std::vector<LineSegment> linesRG = lineSegmentDetection(rgTensorField, itsQuantError, itsAngleTolerance, itsEPS);
  //std::vector<LineSegment> linesBY = lineSegmentDetection(byTensorField, itsQuantError, itsAngleTolerance, itsEPS);


  

  std::vector<Point2D<int> > list_p;
  Image<float> modgrad;

  int max_grad = 260100;
  int n_bins = 16256;
  Image<float> angles = ll_angle(lumTensorField,max_grad*0.10,list_p,modgrad,n_bins,max_grad);


  CpuTimer timer;
  timer.reset();

  WorkThreadServer threadServer("V2", 20);

  std::vector<rutz::shared_ptr<TrainNFAJob> > jobs;
  for(int rw=0; rw<20; rw += 1)
  {
    jobs.push_back(rutz::make_shared(new TrainNFAJob(this, angles,
            itsLocations, rw+3,
            0, itsPDF[rw])));
    threadServer.enqueueJob(jobs.back());
  }

  LINFO("Waiting for jobs to finish");
  while((uint)threadServer.getNumRun() < jobs.size())
  {
    //LINFO("%i/%zu jobs completed", threadServer.getNumRun(),jobs.size());
    sleep(1);
  }
  timer.mark();
  LINFO("Total time for image %0.2f sec", timer.real_secs());

  writePDF(itsPDF, itsTrainNFAFile.getVal().c_str());
}

void V2::cornerSymetryDetection(std::vector<LineSegment>& lines)
{
  itsCornersState.clear();
  itsSymmetries.clear();

  for(uint ls1Idx=0; ls1Idx<itsLines.size(); ls1Idx++)
  {
    for(uint ls2Idx=ls1Idx+1; ls2Idx<itsLines.size(); ls2Idx++)
    {
      //find parallel lines
      /* This code is based on the solution of these two input equations:
       *  Pa = P1 + ua (P2-P1)
       *  Pb = P3 + ub (P4-P3)
       *
       * Where line one is composed of points P1 and P2 and line two is composed
       *  of points P3 and P4.
       *
       * ua/b is the fractional value you can multiple the x and y legs of the
       *  triangle formed by each line to find a point on the line.
       *
       * The two equations can be expanded to their x/y components:
       *  Pa.x = p1.x + ua(p2.x - p1.x)
       *  Pa.y = p1.y + ua(p2.y - p1.y)
       *
       *  Pb.x = p3.x + ub(p4.x - p3.x)
       *  Pb.y = p3.y + ub(p4.y - p3.y)
       *
       * When Pa.x == Pb.x and Pa.y == Pb.y the lines intersect so you can come
       *  up with two equations (one for x and one for y):
       *
       * p1.x + ua(p2.x - p1.x) = p3.x + ub(p4.x - p3.x)
       * p1.y + ua(p2.y - p1.y) = p3.y + ub(p4.y - p3.y)
       *
       * ua and ub can then be individually solved for.  This results in the
       *  equations used in the following code.
       */
      //Finding Intersection of the two lines
      LineSegment& lineSeg1 = itsLines[ls1Idx];
      LineSegment& lineSeg2 = itsLines[ls2Idx];

      double x1 = lineSeg1.p1.i, y1 = lineSeg1.p1.j;
      double x2 = lineSeg1.p2.i, y2 = lineSeg1.p2.j;
      double x3 = lineSeg2.p1.i, y3 = lineSeg2.p1.j;
      double x4 = lineSeg2.p2.i, y4 = lineSeg2.p2.j;

      /*
       * If the denominator for the equations for ua and ub is 0 then the two lines are parallel.
       * If the denominator and numerator for the equations for ua and ub are 0 then the two
       * lines are coincident.
       */
      double den = ((y4 - y3)*(x2 - x1)) - ((x4 - x3)*(y2 - y1) );
      Point2D<float> intersectionPoint;
      if(den == 0.0F)
      {
        ////LINFO("Parallel lines");
        ////LINFO("%0.2fx%0.2f %0.2fx%0.2f     %0.2fx%0.2f %0.2fx%0.2f",
        ////    x1, y1, x2, y2,
        ////    x3, y3, x4, y4);

        ////LINFO("%f %f %f %f",
        ////    x1-x3, y1-y3, x2-x4, y2-y4);

        //float L1 = lineSeg1.length;
        //float L2 = lineSeg2.length;

        ////float xVL = ( (lineSeg1.center.i * L1) + (lineSeg2.center.i * L2) ) / (L1 + L2);
        ////float yVL = ( (lineSeg1.center.j * L1) + (lineSeg2.center.j * L2) ) / (L1 + L2);

        //float xG = ((L1 *(x1 + x2) ) + (L2*(x3 + x4)))/(2*(L1+L2));
        //float yG = ((L1 *(y1 + y2) ) + (L2*(y3 + y4)))/(2*(L1+L2));

        //float ori1 = lineSeg1.ori;
        //float ori2 = lineSeg2.ori;

        //float ang = 0;
        //if ( fabs(ori1 - ori2) <= M_PI/2)
        //  ang = ( (ori1 * L1) + (ori2 * L2) ) / (L1 + L2);
        //else
        //  ang = ( (ori1 * L1) + ((ori2 - (M_PI*ori2/fabs(ori2))) * L2) ) / (L1 + L2);

        //Point2D<int> center((int)xG, (int)yG);

        //Image<PixRGB<byte> > tmp(itsV1EdgesDims, ZEROS);
        //drawCircle(tmp, center, 3, PixRGB<byte>(0,255,255), 2);
        //drawLine(tmp, center, ang, 3000.0F, PixRGB<byte>(255,0,0));
        //drawLine(tmp, (Point2D<int>)lineSeg1.p1, (Point2D<int>)lineSeg1.p2, PixRGB<byte>(255,0,0));
        //drawLine(tmp, (Point2D<int>)lineSeg2.p1, (Point2D<int>)lineSeg2.p2, PixRGB<byte>(0,255,0));
        //SHOWIMG(tmp);

        //drawLine(tmp, (Point2D<int>)lineSeg1.center, lineSeg1.ori, 3000.0F, PixRGB<byte>(255,0,0));
        //drawLine(tmp, (Point2D<int>)lineSeg2.center, lineSeg2.ori, 3000.0F, PixRGB<byte>(0,255,0));
        //SHOWIMG(tmp);
      } else {
        //Find the intersection
        double u_a = (((x4 - x3)*(y1 - y3) ) - (( y4 - y3)*(x1 - x3))) / den;
        double u_b = (((x2 - x1)*(y1 - y3) ) - (( y2 - y1)*(x1 - x3))) / den;

        intersectionPoint.i = x1 + (u_a * (x2 - x1));
        intersectionPoint.j = y1 + (u_a * (y2 - y1));

        if (u_a >= -0.1 && u_a <= 1.1 &&
            u_b >= -0.1 && u_b <= 1.1)
        {

          //LINFO("Line1 Ori %0.2f line 2 Ori %0.2f",
          //    lineSeg1.ori*180/M_PI, lineSeg2.ori*180/M_PI);

          float dotProduct =
            ((lineSeg1.center.i - intersectionPoint.i)*(lineSeg2.center.i - intersectionPoint.i)) +
            ((lineSeg1.center.j - intersectionPoint.j)*(lineSeg2.center.j - intersectionPoint.j));
          float crossProduct =
            ((lineSeg1.center.i - intersectionPoint.i)*(lineSeg2.center.j - intersectionPoint.j)) +
            ((lineSeg1.center.j - intersectionPoint.j)*(lineSeg2.center.i - intersectionPoint.i));

          float ang = atan2(crossProduct, dotProduct);

          float ori1 = atan2((intersectionPoint.j - lineSeg1.center.j),
                            (lineSeg1.center.i - intersectionPoint.i));
          float ori2 = atan2((intersectionPoint.j - lineSeg2.center.j),
                            (lineSeg2.center.i - intersectionPoint.i));

          //Find width direction the angle bisector is facing and set that as the orinetation
          if (ori1 < 0 )
            ori1 += 2*M_PI;
          if (ori2 < 0 )
            ori2 += 2*M_PI;

          float ori = (ori2 - ori1);

          if ( (fabs(ori) > M_PI && ori < 0 ) ||
               (fabs(ori) < M_PI && ori > 0 ) )
            ori = ori1 + fabs(ang/2);
          else
            ori = ori1 - fabs(ang/2);

          //Dont add any sharp or flat corners
          if (fabs(ang) < 170*M_PI/180 && fabs(ang) > 5*M_PI/180)
          {
            CornerState corner(intersectionPoint,
                ori,
                ang);
            corner.lineSeg1 = ls1Idx;
            corner.lineSeg2 = ls2Idx;


            //Find which end points are furthest from the corner
            if (lineSeg1.p1.distance(intersectionPoint) < lineSeg1.p2.distance(intersectionPoint))
              corner.endPoint1 = lineSeg1.p2;
            else
              corner.endPoint1 = lineSeg1.p1;

            if (lineSeg2.p1.distance(intersectionPoint) < lineSeg2.p2.distance(intersectionPoint))
              corner.endPoint2 = lineSeg2.p2;
            else
              corner.endPoint2 = lineSeg2.p1;

            itsCornersState.push_back(corner);
          }

        } else {
          Image<PixRGB<byte> > tmp(320,240, ZEROS);
          if (!tmp.coordsOk(intersectionPoint))
          {
            float L1 = lineSeg1.length;
            float L2 = lineSeg2.length;

            //float xVL = ( (lineSeg1.center.i * L1) + (lineSeg2.center.i * L2) ) / (L1 + L2);
            //float yVL = ( (lineSeg1.center.j * L1) + (lineSeg2.center.j * L2) ) / (L1 + L2);

            float xG = ((L1 *(x1 + x2) ) + (L2*(x3 + x4)))/(2*(L1+L2));
            float yG = ((L1 *(y1 + y2) ) + (L2*(y3 + y4)))/(2*(L1+L2));

            float ori1 = lineSeg1.ori;
            float ori2 = lineSeg2.ori;

            float ang = 0;
            if ( fabs(ori1 - ori2) <= M_PI/2)
              ang = ( (ori1 * L1) + (ori2 * L2) ) / (L1 + L2);
            else
              ang = ( (ori1 * L1) + ((ori2 - (M_PI*ori2/fabs(ori2))) * L2) ) / (L1 + L2);


            Point2D<float> p1Proj((((y1-yG)*sin(ang)) + ( (x1-xG)*cos(ang))),
                                  (((y1-yG)*cos(ang)) - ( (x1-xG)*sin(ang))));
            //Point2D<float> p1Proj( (x1-xG), (y1-yG));

            //Convert P1Proj back to 0 0 origin
            //Point2D<float> p1Disp((((p1Proj.j+yG)*sin(M_PI+ang)) + ( (p1Proj.i+xG)*cos(M_PI+ang))),
            //                      (((p1Proj.j+yG)*cos(M_PI+ang)) - ( (p1Proj.i+xG)*sin(M_PI+ang))));
            //LINFO("POint %fx%f proj %fx%f \n point=(%f,%f)",
            //    x1, y1,
            //    p1Proj.i, p1Proj.j,
            //    p1Disp.i, p1Disp.j);

            Point2D<float> p2Proj(((y2-yG)*sin(ang)) + ( (x2-xG)*cos(ang)),
                                  ((y2-yG)*cos(ang)) - ( (x2-xG)*sin(ang)));
            Point2D<float> p3Proj(((y3-yG)*sin(ang)) + ( (x3-xG)*cos(ang)),
                                  ((y3-yG)*cos(ang)) - ( (x3-xG)*sin(ang)));
            Point2D<float> p4Proj(((y4-yG)*sin(ang)) + ( (x4-xG)*cos(ang)),
                                  ((y4-yG)*cos(ang)) - ( (x4-xG)*sin(ang)));

            float points[4] = {p1Proj.i, p2Proj.i, p3Proj.i, p4Proj.i};

            float min = points[0];
            float max = points[0];

            for(int i=1; i<4; i++)
            {
              if (points[i] > max)
                max = points[i];
              if (points[i] < min)
                min = points[i];
            }

            //LINFO("POints %f %f %f %f",
            //    points[0], points[1], points[2], points[3]);
            //LINFO("Max %f min %f", max, min);

            float totalLength = fabs(max-min);
            //LINFO("Distance %f  totalLength %f",
            //    totalLength, L1+L2);

            //Do we have overlap with some noise
            if (totalLength < L1+L2 + 10)
            {
              SymmetryState  symState;
              symState.lineSeg1 = ls1Idx;
              symState.lineSeg2 = ls2Idx;
              symState.center = Point2D<float>(xG,yG);
              symState.ang = ang;
              symState.length = totalLength;

              itsSymmetries.push_back(symState);


             // Point2D<int> center((int)xG, (int)yG);

             // //drawCircle(tmp, (Point2D<int>)p1Disp, 3, PixRGB<byte>(0,255,0), 2);
             // //drawCircle(tmp, (Point2D<int>)p2Proj, 3, PixRGB<byte>(0,255,0), 2);
             // //drawCircle(tmp, (Point2D<int>)p3Proj, 3, PixRGB<byte>(0,255,0), 2);
             // //drawCircle(tmp, (Point2D<int>)p4Proj, 3, PixRGB<byte>(0,255,0), 2);

             // drawCircle(tmp, center, 3, PixRGB<byte>(0,255,255), 2);
             // drawLine(tmp, center, ang, 3000.0F, PixRGB<byte>(255,0,0));
             // drawLine(tmp, (Point2D<int>)lineSeg1.p1, (Point2D<int>)lineSeg1.p2, PixRGB<byte>(255,0,0));
             // drawLine(tmp, (Point2D<int>)lineSeg2.p1, (Point2D<int>)lineSeg2.p2, PixRGB<byte>(0,255,0));
             // SHOWIMG(tmp);

             // drawLine(tmp, (Point2D<int>)lineSeg1.center, lineSeg1.ori, 3000.0F, PixRGB<byte>(255,0,0));
             // drawLine(tmp, (Point2D<int>)lineSeg2.center, lineSeg2.ori, 3000.0F, PixRGB<byte>(0,255,0));
             // SHOWIMG(tmp);
            }
          }
        }
      }



      //LINFO("Intersection %fx%f", x, y);


      //if (!tmpImg.coordsOk(intersectionPoint) &&
      //    fabs(lineSeg1.length - lineSeg2.length) < 10.0F && false)
      //{
      //  Image<PixRGB<byte> > tmp = tmpImg;
      //  drawLine(tmp, (Point2D<int>)lineSeg1.p1, (Point2D<int>)lineSeg1.p2, PixRGB<byte>(255,0,0));
      //  drawLine(tmp, (Point2D<int>)lineSeg2.p1, (Point2D<int>)lineSeg2.p2, PixRGB<byte>(0,255,0));
      //  SHOWIMG(tmp);

      //  drawLine(tmp, (Point2D<int>)lineSeg1.center, lineSeg1.ori, 3000.0F, PixRGB<byte>(255,0,0));
      //  drawLine(tmp, (Point2D<int>)lineSeg2.center, lineSeg2.ori, 3000.0F, PixRGB<byte>(0,255,0));
      //  drawCircle(tmp, (Point2D<int>)intersectionPoint, 6, PixRGB<byte>(0,255,255));
      //  SHOWIMG(tmp);
      //}


    }
  }

}





void V2::findLines(const Image<float> &mag, const Image<float>& ori, Image<float>& retMag, Image<float>& retOri)
{
  retMag = Image<float>(mag.getDims()/2, ZEROS);
  retOri = Image<float>(ori.getDims()/2, ZEROS);

  Dims win(3,3);
  for(int y=0; y<mag.getHeight()-win.h()-1; y += 2)
    for(int x=0; x<mag.getWidth()-win.w()-1; x += 2)
    {
      double sumSin = 0, sumCos = 0;
      float numElem = 0;
      float totalWeight = 0;

      double meanLocX = 0, meanLocY = 0;
      Image<float> tt(win, ZEROS);

      //Add the angles within the window into the histogram
      float hist[360];
      for(int wy=0; wy<win.h(); wy++)
        for(int wx=0; wx<win.w(); wx++)
        {
          float val = mag.getVal(x+wx,y+wy);

          if (val > 0)
          {
            float ang = ori.getVal(x+wx,y+wy) + M_PI/2;
            meanLocX += wx;
            meanLocY += wy;
            sumSin += sin(ang); //*mag;
            sumCos += cos(ang); //*mag;
            numElem++;
            totalWeight += 1; //mag;

            //Add to histogram
            int angI = (int)(ang*180.0/M_PI);
            if (angI < 0) angI += 360;
            if (angI > 360) angI -= 360;
            hist[angI] += 1; //mag;
            //tt.setVal(wx,wy,mag);
          }
        }
      if (numElem > 0)
      {
        double xMean = sumSin/totalWeight;
        double yMean = sumCos/totalWeight;
        double sum = sqrt( xMean*xMean + yMean*yMean);
        float ang = atan2(xMean, yMean);

        if (sum > 0.99)
        {
          //This must be an edge, calculate the angle from the mean
          //drawLine(LTmp, Point2D<int>(x+(int)(meanLocX/numElem), y+(int)(meanLocY/numElem)), (float)ang, win.w(), 255.0F); // (float)ang);
          //V1::EdgeState edgeState;
          //edgeState.pos = Point2D<int>(x+(int)(meanLocX/numElem), y+(int)(meanLocY/numElem));
          //edgeState.ori = ang;
          //edgeState.var = (10*M_PI/180)*(10*M_PI/180); //10
          //edgeState.prob = sum;
          //itsEdgesState.push_back(edgeState);

          retMag.setVal(Point2D<int>(x/2+(int)(meanLocX/numElem), y/2+(int)(meanLocY/numElem)), 255.0);
          retOri.setVal(Point2D<int>(x/2+(int)(meanLocX/numElem), y/2+(int)(meanLocY/numElem)), ang-M_PI/2);
          //nLines++;
        } else {
          //We have a corner
        }
      } else {
        //Stmp.setVal(x,y,0.0);
      }
    }

}

void V2::evolve(Image<PixRGB<byte> >& img)
{

  Image<float> lum = luminance(img);
  SHOWIMG(lum);
  inplaceNormalize(lum, 0.0F, 255.0F);

  std::vector<LineSegment> lines = lineSegmentDetection(lum);

  Image<PixRGB<byte> > edgesImg(lum.getDims(), ZEROS);
  for(uint i=0; i<lines.size(); i++)
  {
    LineSegment& ls = lines[i];
    drawLine(edgesImg, (Point2D<int>)ls.p1, (Point2D<int>)ls.p2, PixRGB<byte>(255,0,0), 1); //(int)ls.width);
  }

  SHOWIMG(edgesImg);

}

void V2::evolve(Image<float>& img)
{

  //SHOWIMG(img);
  inplaceNormalize(img, 0.0F, 255.0F);

  itsLines = lineSegmentDetection(img);

  //Image<PixRGB<byte> > edgesImg(.getDims(), ZEROS);
  //for(uint i=0; i<lines.size(); i++)
  //{
  //  LineSegment& ls = lines[i];
  //  drawLine(edgesImg, (Point2D<int>)ls.p1, (Point2D<int>)ls.p2, PixRGB<byte>(255,0,0), 1); //(int)ls.width);
  //}

  //SHOWIMG(edgesImg);

}


void V2::evolve(TensorField& tensorField)
{
  nonMaxSurp(tensorField);

  LINFO("Get LInes");
  std::vector<LineSegment> lines = lineSegmentDetection(tensorField);
  LINFO("Show %i Lines", (uint)lines.size());

  Image<PixRGB<byte> > edgesImg(tensorField.t1.getDims(), ZEROS);
  for(uint i=0; i<lines.size(); i++)
  {
    LineSegment& ls = lines[i];
    drawLine(edgesImg, (Point2D<int>)ls.p1, (Point2D<int>)ls.p2, PixRGB<byte>(255,0,0), 1); //(int)ls.width);
  }

  //SHOWIMG(edgesImg);

}

Layout<PixRGB<byte> > V2::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  Image<float> input = itsLGNInput[0];
  //input = rescale(input, 320, 240);
  Image<PixRGB<byte> > linesLumImg = input;
  Image<PixRGB<byte> > linesImg(input.getDims(), ZEROS);
  for(uint i=0; i<itsLines.size(); i++)
  {
    LineSegment& ls = itsLines[i];
    if (ls.strength > 0)
    {
      PixRGB<byte> color;
      switch(ls.color)
      {
        case 0: color = PixRGB<byte>(255,0,0); break;
        case 1: color = PixRGB<byte>(0,255,0); break;
        case 2: color = PixRGB<byte>(0,0,255); break;
      }

      drawLine(linesLumImg, (Point2D<int>)ls.p1, (Point2D<int>)ls.p2,color, 1); //(int)ls.width);
      drawLine(linesImg, (Point2D<int>)ls.p1, (Point2D<int>)ls.p2, PixRGB<byte>(255,0,0)); 
    }
  }
  outDisp = hcat(linesLumImg, linesImg);
  EigenSpace eigen = getTensorEigen(itsMaxTF);
  Image<float> maxTf = eigen.l1-eigen.l2;
  inplaceNormalize(maxTf, 0.0F, 255.0F);
  Image<PixRGB<byte> > maxEdges = toRGB(maxTf);
  outDisp = hcat(outDisp, maxEdges);

  Layout<PixRGB<byte> > tensorDisp;
  Image<PixRGB<byte> > lumEdges = toRGB(itsLumTV.getTokensMag(true));
  Image<PixRGB<byte> > rgEdges = toRGB(itsRGTV.getTokensMag(true));
  Image<PixRGB<byte> > byEdges = toRGB(itsBYTV.getTokensMag(true));

  tensorDisp = hcat(lumEdges, rgEdges);
  tensorDisp = hcat(tensorDisp, byEdges);

  outDisp = vcat(outDisp, tensorDisp);

  return outDisp;

}



///////////////////////////////////////////////////////////////////////// LSD computations ////////////////////////////////////////////////////
/*----------------------------------------------------------------------------*/
/*------------------------------ Gradient Angle ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*
   compute the direction of the level line at each point.
   it returns:

   - an image_float with the angle at each pixel or NOTDEF.
   - the image_float 'modgrad' (a pointer is passed as argument)
     with the gradient magnitude at each point.
   - a list of pixels 'list_p' roughly ordered by gradient magnitude.
     (the order is made by classing points into bins by gradient magnitude.
      the parameters 'n_bins' and 'max_grad' specify the number of
      bins and the gradient modulus at the highest bin.)
   - a pointer 'mem_p' to the memory used by 'list_p' to be able to
     free the memory.
 */
Image<float> V2::ll_angle(Image<float>& in, float threshold,
                             std::vector<Point2D<int> > &list_p, void ** mem_p,
                             Image<float>& modgrad, int n_bins, int max_grad )
{
  Image<float> grad(in.getWidth(), in.getHeight(), ZEROS);
  int n,p,adr,i;
  float com1,com2,gx,gy,norm2;
  /* variables used in pseudo-ordering of gradient magnitude */
  float f_n_bins = (float) n_bins;
  float f_max_grad = (float) max_grad;
  int list_count = 0;
  struct coorlist * list;
  struct coorlist ** range_l_s;
  struct coorlist ** range_l_e;
  struct coorlist * start;
  struct coorlist * end;


  threshold *= 4.0 * threshold;

  n = in.getHeight();
  p = in.getWidth();

  /* get memory for the image of gradient modulus */
  modgrad = Image<float>(in.getWidth(),in.getHeight(), ZEROS);

  /* get memory for "ordered" coordinate list */
  list = (struct coorlist *) calloc(n*p,sizeof(struct coorlist));
  *mem_p = (void *) list;
  range_l_s = (struct coorlist **) calloc(n_bins,sizeof(struct coorlist *));
  range_l_e = (struct coorlist **) calloc(n_bins,sizeof(struct coorlist *));
  if( !list || !range_l_s || !range_l_e ) LFATAL("Not enough memory.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;

  /* undefined on downright boundary */
  for(int x=0;x<p;x++) grad[(n-1)*p+x] = NOTDEF;
  for(int y=0;y<n;y++) grad[p*y+p-1]   = NOTDEF;

  /*** remaining part ***/
  for(int x=0;x<p-1;x++)
    for(int y=0;y<n-1;y++)
      {
        adr = y*p+x;

        /* norm 2 computation */
        com1 = in[adr+p+1] - in[adr];
        com2 = in[adr+1]   - in[adr+p];
        gx = com1+com2;
        gy = com1-com2;
        norm2 = gx*gx+gy*gy;

        modgrad[adr] = (float) sqrt( (double) norm2 / 4.0 );

        if(norm2 <= threshold) /* norm too small, gradient no defined */
          grad[adr] = NOTDEF;
        else
          {
            /* angle computation */
            grad[adr] = (float) atan2(gx,-gy);

            /* store the point in the right bin,
               according to its norm */
            i = (int) (norm2 * f_n_bins / f_max_grad);
            if(i>=n_bins) i = n_bins-1;
            if( range_l_e[i]==NULL )
              range_l_s[i] = range_l_e[i] = list+list_count++;
            else
              {
                range_l_e[i]->next = list+list_count;
                range_l_e[i] = list+list_count++;
              }
            range_l_e[i]->x = x;
            range_l_e[i]->y = y;
            range_l_e[i]->next = NULL;
          }
      }

  /* make the list of points "ordered" by norm value */
  for(i=n_bins-1; i>0 && range_l_s[i]==NULL; i--);
  start = range_l_s[i];
  end = range_l_e[i];
  if(start!=NULL)
    for(i--;i>0; i--)
      if( range_l_s[i] != NULL )
        {
          end->next = range_l_s[i];
          end = range_l_e[i];
        }

  struct coorlist * lp = start;

  for(;lp; lp = lp->next )
    list_p.push_back(Point2D<int>(lp->x, lp->y));


  /* free memory */
  free(range_l_s);
  free(range_l_e);

  return grad;
}

/*----------------------------------------------------------------------------*/
/*------------------------------ Gradient Angle ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*
   compute the direction of the level line at each point.
   it returns:

   - an image_float with the angle at each pixel or NOTDEF.
   - the image_float 'modgrad' (a pointer is passed as argument)
     with the gradient magnitude at each point.
   - a list of pixels 'list_p' roughly ordered by gradient magnitude.
     (the order is made by classing points into bins by gradient magnitude.
      the parameters 'n_bins' and 'max_grad' specify the number of
      bins and the gradient modulus at the highest bin.)
   - a pointer 'mem_p' to the memory used by 'list_p' to be able to
     free the memory.
 */
Image<float> V2::ll_angle(const TensorField& tensorField, float threshold,
                             std::vector<Point2D<int> > &list_p,
                             Image<float>& modgrad, int n_bins, int max_grad )
{
  Image<float> grad(tensorField.t1.getWidth(), tensorField.t1.getHeight(), ZEROS);
  int n,p,i;
  //float com1,com2,gx,gy,norm2;
  /* variables used in pseudo-ordering of gradient magnitude */
  float f_n_bins = (float) n_bins;
  float f_max_grad = (float) max_grad;
  int list_count = 0;
  struct coorlist * list;
  struct coorlist ** range_l_s;
  struct coorlist ** range_l_e;
  struct coorlist * start;
  struct coorlist * end;


  n = tensorField.t1.getHeight();
  p = tensorField.t1.getWidth();

  /* get memory for the image of gradient modulus */
  modgrad = Image<float>(tensorField.t1.getWidth(),tensorField.t1.getHeight(), ZEROS);

  /* get memory for "ordered" coordinate list */
  list = (struct coorlist *) calloc(n*p,sizeof(struct coorlist));
  range_l_s = (struct coorlist **) calloc(n_bins,sizeof(struct coorlist *));
  range_l_e = (struct coorlist **) calloc(n_bins,sizeof(struct coorlist *));
  if( !list || !range_l_s || !range_l_e ) LFATAL("Not enough memory.");

  for(i=0;i<n_bins;i++)
    range_l_s[i] = range_l_e[i] = NULL;

  EigenSpace eigen = getTensorEigen(tensorField);
  Image<float> features = eigen.l1-eigen.l2;
  inplaceNormalize(features, 0.0F, f_max_grad);

  for(int y=0; y<features.getHeight(); y++)
    for(int x=0; x<features.getWidth(); x++)
    {
      /* norm 2 computation */
      //double norm2 = tensorField.t1.getVal(x,y) + tensorField.t4.getVal(x,y);
      //modgrad.setVal(x,y, (float) sqrt( (double) norm2 / 4.0 ));
      //float norm2 = eigen.l1.getVal(x,y) - eigen.l2.getVal(x,y);
      float val = features.getVal(x,y);
      modgrad.setVal(x,y, val);

      if(val > threshold) /* norm too small, gradient no defined */
      {
        /* angle computation */

        //Get the direction of the vote from e1, while the weight is l1-l2
        float u = eigen.e1[1].getVal(x,y);
        float v = eigen.e1[0].getVal(x,y);
        grad.setVal(x,y, atan(-u/v));

        /* store the point in the right bin,
           according to its norm */
        i = (int) (val * f_n_bins / f_max_grad);
        if(i>=n_bins) i = n_bins-1;
        if( range_l_e[i]==NULL )
          range_l_s[i] = range_l_e[i] = list+list_count++;
        else
        {
          range_l_e[i]->next = list+list_count;
          range_l_e[i] = list+list_count++;
        }
        range_l_e[i]->x = x;
        range_l_e[i]->y = y;
        range_l_e[i]->next = NULL;
      } else {
        grad.setVal(x,y,NOTDEF);
      }
  }

  /* make the list of points "ordered" by norm value */
  for(i=n_bins-1; i>0 && range_l_s[i]==NULL; i--);
  start = range_l_s[i];
  end = range_l_e[i];
  if(start!=NULL)
    for(i--;i>0; i--)
      if( range_l_s[i] != NULL )
        {
          end->next = range_l_s[i];
          end = range_l_e[i];
        }

  struct coorlist * lp = start;

  for(;lp; lp = lp->next )
    list_p.push_back(Point2D<int>(lp->x, lp->y));

  /* free memory */
  free(range_l_s);
  free(range_l_e);
  free(list);

  return grad;
}

/*----------------------------------------------------------------------------*/
/*
   find if the point x,y in angles have angle theta up to precision prec
 */
bool V2::isaligned(Point2D<int> loc,const Image<float>& angles,
                   float theta, float prec)
{
  if (!angles.coordsOk(loc)) return false;

  float a = angles.getVal(loc);

  if( a == NOTDEF ) return false;

  /* it is assumed that theta and a are in the range [-pi,pi] */
  theta -= a;
  if( theta < 0.0 ) theta = -theta;
  if( theta > M_3_2_PI )
    {
      theta -= M_2__PI;
      if( theta < 0.0 ) theta = -theta;
    }

  //LINFO("Theta %f < %f", theta, prec);
  return theta < prec;
}

/*----------------------------------------------------------------------------*/
float V2::angle_diff(float a, float b)
{
  a -= b;
  while( a <= -M_PI ) a += 2.0*M_PI;
  while( a >   M_PI ) a -= 2.0*M_PI;
  if( a < 0.0 ) a = -a;
  return a;
}



/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*
   Calculates the natural logarithm of the absolute value of
   the gamma function of x using the Lanczos approximation,
   see http://www.rskey.org/gamma.htm.

   The formula used is
     \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                 (x+5.5)^(x+0.5) e^{-(x+5.5)}
   so
     \log\Gamma(x) = \log( \sum_{n=0}^{N} q_n x^n ) + (x+0.5) \log(x+5.5)
                     - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
   and
     q0 = 75122.6331530
     q1 = 80916.6278952
     q2 = 36308.2951477
     q3 = 8687.24529705
     q4 = 1168.92649479
     q5 = 83.8676043424
     q6 = 2.50662827511
 */
double V2::log_gamma_lanczos(double x)
{
  static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
  double a = (x+0.5) * log(x+5.5) - (x+5.5);
  double b = 0.0;
  int n;

  for(n=0;n<7;n++)
    {
      a -= log( x + (double) n );
      b += q[n] * pow( x, (double) n );
    }
  return a + log(b);
}

/*----------------------------------------------------------------------------*/
/*
   Calculates the natural logarithm of the absolute value of
   the gamma function of x using Robert H. Windschitl method,
   see http://www.rskey.org/gamma.htm.

   The formula used is
     \Gamma(x) = \sqrt(\frac{2\pi}{x}) ( \frac{x}{e}
                   \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } )^x
   so
     \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                     + 0.5x\log( x\sinh(1/x) + \frac{1}{810x^6} ).

   This formula is good approximation when x > 15.
 */
double V2::log_gamma_windschitl(double x)
{
  return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*----------------------------------------------------------------------------*/
/*
   Calculates the natural logarithm of the absolute value of
   the gamma function of x. When x>15 use log_gamma_windschitl(),
   otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/*
   Computes the logarithm of NFA to base 10.

   NFA = NT.b(n,k,p)
   the return value is log10(NFA)

   n,k,p - binomial parameters.
   logNT - logarithm of Number of Tests
 */
#define TABSIZE 100000
double V2::nfa(int n, int k, double p, double logNT)
{
  static double inv[TABSIZE];   /* table to keep computed inverse values */
  double tolerance = 0.1;       /* an error of 10% in the result is accepted */
  double log1term,term,bin_term,mult_term,bin_tail,err;
  double p_term = p / (1.0-p);
  int i;

  if( n<0 || k<0 || k>n || p<0.0 || p>1.0 )
    LFATAL("Wrong n=%i, k=%i or p=%f values in nfa()", n, k, p);

  if( n==0 || k==0 ) return -logNT;
  if( n==k ) return -logNT - (double) n * log10(p);

  log1term = log_gamma((double)n+1.0) - log_gamma((double)k+1.0)
           - log_gamma((double)(n-k)+1.0)
           + (double) k * log(p) + (double) (n-k) * log(1.0-p);

  term = exp(log1term);
  if( double_equal(term,0.0) )              /* the first term is almost zero */
    {
      if( (double) k > (double) n * p )    /* at begin or end of the tail? */
        return -log1term / M_LN10 - logNT; /* end: use just the first term */
      else
        return -logNT;                     /* begin: the tail is roughly 1 */
    }

  bin_tail = term;
  for(i=k+1;i<=n;i++)
    {
      bin_term = (double) (n-i+1) * ( i<TABSIZE ?
                   ( inv[i] != 0.0 ? inv[i] : (inv[i]=1.0/(double)i))
                   : 1.0/(double)i );
      mult_term = bin_term * p_term;
      term *= mult_term;
      bin_tail += term;
      if(bin_term<1.0)
        {
          /* when bin_term<1 then mult_term_j<mult_term_i for j>i.
             then, the error on the binomial tail when truncated at
             the i term can be bounded by a geometric series of form
             term_i * sum mult_term_i^j.                            */
          err = term * ( ( 1.0 - pow( mult_term, (double) (n-i+1) ) ) /
                         (1.0-mult_term) - 1.0 );

          /* one wants an error at most of tolerance*final_result, or:
             tolerance * abs(-log10(bin_tail)-logNT).
             now, the error that can be accepted on bin_tail is
             given by tolerance*final_result divided by the derivative
             of -log10(x) when x=bin_tail. that is:
             tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
             finally, we truncate the tail if the error is less than:
             tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
          if( err < tolerance * fabs(-log10(bin_tail)-logNT) * bin_tail ) break;
        }
    }
  return -log10(bin_tail) - logNT;
}



//double V2::nfa(int n, int k, double p, double logNT)
//{
//  double N1[2] = {1.0337965773779e-06,   -0.00215083485137987};
//  double N2[2] = {-8.53870284101796e-06,  0.0273185078224109};
//  double N3[2] = {0.000230591970586356,  -0.828902881766958};
//  double N4[2] = {-0.000233656573018064,  0.719832739194112};
//
//  double F1 = N1[1]*n + N1[0];
//  double F2 = N2[1]*n + N2[0];
//  double F3 = N3[1]*n + N3[0];
//  double F4 = N4[1]*n + N4[0];
//
//
//  double prob = F1*k*k*k + F2*k*k + F3*k + F4;
//
//  return -log10(exp(prob)) - logNT;
//}



/*----------------------------------------------------------------------------*/
void V2::rect_copy(struct rect * in, struct rect * out)
{
  out->x1 = in->x1;
  out->y1 = in->y1;
  out->x2 = in->x2;
  out->y2 = in->y2;
  out->width = in->width;
  out->x = in->x;
  out->y = in->y;
  out->theta = in->theta;
  out->dx = in->dx;
  out->dy = in->dy;
  out->prec = in->prec;
  out->p = in->p;
}



/*----------------------------------------------------------------------------*/
float V2::inter_low(float x, float x1, float y1, float x2, float y2)
{
  if( x1 > x2 || x < x1 || x > x2 )
    {
      LFATAL("inter_low: x %g x1 %g x2 %g.\n",x,x1,x2);
      LFATAL("Impossible situation.");
    }
  if( x1 == x2 && y1<y2 ) return y1;
  if( x1 == x2 && y1>y2 ) return y2;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
float V2::inter_hi(float x, float x1, float y1, float x2, float y2)
{
  if( x1 > x2 || x < x1 || x > x2 )
    {
      LFATAL("inter_hi: x %g x1 %g x2 %g.\n",x,x1,x2);
      LFATAL("Impossible situation.");
    }
  if( x1 == x2 && y1<y2 ) return y2;
  if( x1 == x2 && y1>y2 ) return y1;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
void V2::ri_del(rect_iter * iter)
{
  free(iter);
}

/*----------------------------------------------------------------------------*/
int V2::ri_end(rect_iter * i)
{
  return (float)(i->x) > i->vx[2];
}

int V2::ri_end(rect_iter& itr)
{
  return (float)(itr.x) > itr.vx[2];
}

/*----------------------------------------------------------------------------*/
void V2::ri_inc(rect_iter * i)
{
  if( (float) (i->x) <= i->vx[2] ) i->y++;

  while( (float) (i->y) > i->ye && (float) (i->x) <= i->vx[2] )
    {
      /* new x */
      i->x++;

      if( (float) (i->x) > i->vx[2] ) return; /* end of iteration */

      /* update lower y limit for the line */
      if( (float) i->x < i->vx[3] )
        i->ys = inter_low((float)i->x,i->vx[0],i->vy[0],i->vx[3],i->vy[3]);
      else i->ys = inter_low((float)i->x,i->vx[3],i->vy[3],i->vx[2],i->vy[2]);

      /* update upper y limit for the line */
      if( (float)i->x < i->vx[1] )
        i->ye = inter_hi((float)i->x,i->vx[0],i->vy[0],i->vx[1],i->vy[1]);
      else i->ye = inter_hi( (float)i->x,i->vx[1],i->vy[1],i->vx[2],i->vy[2]);

      /* new y */
      i->y = (int)((float) ceil( (double) i->ys ));
    }
}

/*----------------------------------------------------------------------------*/
void V2::ri_inc(rect_iter& itr)
{
  if( (float) (itr.x) <= itr.vx[2] ) itr.y++;

  while( (float) (itr.y) > itr.ye && (float) (itr.x) <= itr.vx[2] )
    {
      /* new x */
      itr.x++;

      if( (float) (itr.x) > itr.vx[2] ) return; /* end of iteration */

      /* update lower y limit for the line */
      if( (float) itr.x < itr.vx[3] )
        itr.ys = inter_low((float)itr.x,itr.vx[0],itr.vy[0],itr.vx[3],itr.vy[3]);
      else itr.ys = inter_low((float)itr.x,itr.vx[3],itr.vy[3],itr.vx[2],itr.vy[2]);

      /* update upper y limit for the line */
      if( (float)itr.x < itr.vx[1] )
        itr.ye = inter_hi((float)itr.x,itr.vx[0],itr.vy[0],itr.vx[1],itr.vy[1]);
      else itr.ye = inter_hi( (float)itr.x,itr.vx[1],itr.vy[1],itr.vx[2],itr.vy[2]);

      /* new y */
      itr.y = (int)((float) ceil( (double) itr.ys ));
    }
}


/*----------------------------------------------------------------------------*/
V2::rect_iter * V2::ri_ini(struct rect * r)
{
  float vx[4],vy[4];
  int n,offset;
  rect_iter * i;

  i = (rect_iter *) malloc(sizeof(rect_iter));
  if(!i) LFATAL("ri_ini: Not enough memory.");

  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;
  /* else if( r->x1 <= r->x2 && r->y1 > r->y2 ) offset = 3; */

  for(n=0; n<4; n++)
    {
      i->vx[n] = vx[(offset+n)%4];
      i->vy[n] = vy[(offset+n)%4];
    }

  /* starting point */
  i->x = (int)(ceil( (double) (i->vx[0]) ) - 1);
  i->y = (int)(ceil( (double) (i->vy[0]) ));
  i->ys = i->ye = -BIG_NUMBER;

  /* advance to the first point */
  ri_inc(i);

  return i;
}

/*----------------------------------------------------------------------------*/
void V2::ri_ini(struct rect * r, rect_iter& itr)
{
  float vx[4],vy[4];
  int n,offset;

  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;
  /* else if( r->x1 <= r->x2 && r->y1 > r->y2 ) offset = 3; */

  for(n=0; n<4; n++)
    {
      itr.vx[n] = vx[(offset+n)%4];
      itr.vy[n] = vy[(offset+n)%4];
    }

  /* starting point */
  itr.x = (int)(ceil( (double) (itr.vx[0]) ) - 1);
  itr.y = (int)(ceil( (double) (itr.vy[0]) ));
  itr.ys = itr.ye = -BIG_NUMBER;

  /* advance to the first point */
  ri_inc(itr);
}

/*----------------------------------------------------------------------------*/
double V2::rect_nfa(struct rect * rec, Image<float>& angles, double logNT)
{
  rect_iter * i;
  int pts = 0;
  int alg = 0;
  double nfa_val;

  for(i=ri_ini(rec); !ri_end(i); ri_inc(i))
    if( i->x>=0 && i->y>=0 && i->x<angles.getWidth() && i->y<angles.getHeight() )
    {
      if(itsLSDVerbose) LINFO("| %d %d ",i->x,i->y);
      ++pts;
      if( isaligned(Point2D<int>(i->x,i->y),angles,rec->theta,rec->prec) )
        ++alg;
    }
  ri_del(i);

  nfa_val = nfa(pts,alg,rec->p,logNT);
  if(itsLSDVerbose) LINFO("\npts %d alg %d p %g nfa %g\n",
                      pts,alg,rec->p,nfa_val);

  return nfa_val;
}


/*----------------------------------------------------------------------------*/
/*---------------------------------- REGIONS ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
float V2::get_theta( struct point * reg, int reg_size, float x, float y,
                        Image<float>& modgrad, float reg_angle, float prec,
                        float * elongation )
{
  int i;
  float Ixx = 0.0;
  float Iyy = 0.0;
  float Ixy = 0.0;
  float lambda1,lambda2,tmp;
  float theta;
  float weight,sum;

  if(reg_size <= 1) LFATAL("get_theta: region size <= 1.");


  /*----------- theta ---------------------------------------------------*/
  /*
      Region inertia matrix A:
         Ixx Ixy
         Ixy Iyy
      where
        Ixx = \sum_i y_i^2
        Iyy = \sum_i x_i^2
        Ixy = -\sum_i x_i y_i

      lambda1 and lambda2 are the eigenvalues, with lambda1 >= lambda2.
      They are found by solving the characteristic polynomial
      det(\lambda I - A) = 0.

      To get the line segment direction we want to get the eigenvector of
      the smaller eigenvalue. We have to solve a,b in:
        a.Ixx + b.Ixy = a.lambda2
        a.Ixy + b.Iyy = b.lambda2
      We want the angle theta = atan(b/a). I can be computed with
      any of the two equations:
        theta = atan( (lambda2-Ixx) / Ixy )
      or
        theta = atan( Ixy / (lambda2-Iyy) )

      When |Ixx| > |Iyy| we use the first, otherwise the second
      (just to get better numeric precision).
   */
  sum = 0.0;
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad[ reg[i].x + reg[i].y * modgrad.getWidth() ];
      Ixx += ((float)reg[i].y - y) * ((float)reg[i].y - y) * weight;
      Iyy += ((float)reg[i].x - x) * ((float)reg[i].x - x) * weight;
      Ixy -= ((float)reg[i].x - x) * ((float)reg[i].y - y) * weight;
      sum += weight;
    }
  if( sum <= 0.0 ) LFATAL("get_theta: weights sum less or equal to zero.");
  Ixx /= sum;
  Iyy /= sum;
  Ixy /= sum;
  lambda1 = ( Ixx + Iyy + (float) sqrt( (double) (Ixx - Iyy) * (Ixx - Iyy)
                                        + 4.0 * Ixy * Ixy ) ) / 2.0;
  lambda2 = ( Ixx + Iyy - (float) sqrt( (double) (Ixx - Iyy) * (Ixx - Iyy)
                                        + 4.0 * Ixy * Ixy ) ) / 2.0;
  if( fabs(lambda1) < fabs(lambda2) )
    {
      fprintf(stderr,"Ixx %g Iyy %g Ixy %g lamb1 %g lamb2 %g - lamb1 < lamb2\n",
                      Ixx,Iyy,Ixy,lambda1,lambda2);
      tmp = lambda1;
      lambda1 = lambda2;
      lambda2 = tmp;
    }
  if(itsLSDVerbose) LINFO("Ixx %g Iyy %g Ixy %g lamb1 %g lamb2 %g\n",
                      Ixx,Iyy,Ixy,lambda1,lambda2);

  *elongation = lambda1/lambda2;

  if( fabs(Ixx) > fabs(Iyy) )
    theta = (float) atan2( (double) lambda2 - Ixx, (double) Ixy );
  else
    theta = (float) atan2( (double) Ixy, (double) lambda2 - Iyy );

  /* the previous procedure don't cares orientations,
     so it could be wrong by 180 degrees.
     here is corrected if necessary */
  if( angle_diff(theta,reg_angle) > prec ) theta += M_PI;

  return theta;
}

/*----------------------------------------------------------------------------*/
float V2::region2rect( struct point * reg, int reg_size,
                          Image<float>& modgrad, float reg_angle,
                          float prec, double p, struct rect * rec,
                          float* sum_l, float* sum_w, int sum_offset, int sum_res)
{
  float x,y,dx,dy,l,w,lf,lb,wl,wr,theta,weight,sum,sum_th,s,elongation;
  int i,n;
  int l_min,l_max,w_min,w_max;

  /* center */
  x = y = sum = 0.0;
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad[ reg[i].x + reg[i].y * modgrad.getWidth() ];
      x += (float) reg[i].x * weight;
      y += (float) reg[i].y * weight;
      sum += weight;
    }
  if( sum <= 0.0 ) LFATAL("region2rect: weights sum equal to zero.");
  x /= sum;
  y /= sum;
  if(itsLSDVerbose) LINFO("center x %g y %g\n",x,y);

  /* theta */
  theta = get_theta(reg,reg_size,x,y,modgrad,reg_angle,prec,&elongation);
  if(itsLSDVerbose) LINFO("theta %g\n",theta);

  /* length and width */
  lf = lb = wl = wr = 0.5;
  l_min = l_max = w_min = w_max = 0;
  dx = (float) cos( (double) theta );
  dy = (float) sin( (double) theta );
  for(i=0; i<reg_size; i++)
    {
      l =  ((float)reg[i].x - x) * dx + ((float)reg[i].y - y) * dy;
      w = -((float)reg[i].x - x) * dy + ((float)reg[i].y - y) * dx;
      weight = modgrad[ reg[i].x + reg[i].y * modgrad.getWidth() ];

      n = (int) MY_ROUND( l * (float) sum_res );
      if(n<l_min) l_min = n;
      if(n>l_max) l_max = n;
      sum_l[sum_offset + n] += weight;

      n = (int) MY_ROUND( w * (float) sum_res );
      if(n<w_min) w_min = n;
      if(n>w_max) w_max = n;
      sum_w[sum_offset + n] += weight;
    }

  sum_th = 0.01 * sum; /* weight threshold for selecting region */
  for(s=0.0,i=l_min; s<sum_th && i<=l_max; i++) s += sum_l[sum_offset + i];
  lb = ( (float) (i-1) - 0.5 ) / (float) sum_res;
  for(s=0.0,i=l_max; s<sum_th && i>=l_min; i--) s += sum_l[sum_offset + i];
  lf = ( (float) (i+1) + 0.5 ) / (float) sum_res;

  sum_th = 0.01 * sum; /* weight threshold for selecting region */
  for(s=0.0,i=w_min; s<sum_th && i<=w_max; i++) s += sum_w[sum_offset + i];
  wr = ( (float) (i-1) - 0.5 ) / (float) sum_res;
  for(s=0.0,i=w_max; s<sum_th && i>=w_min; i--) s += sum_w[sum_offset + i];
  wl = ( (float) (i+1) + 0.5 ) / (float) sum_res;

  if(itsLSDVerbose) LINFO("lb %g lf %g wr %g wl %g\n",lb,lf,wr,wl);

  /* free vector */
  for(i=l_min; i<=l_max; i++) sum_l[sum_offset + i] = 0.0;
  for(i=w_min; i<=w_max; i++) sum_w[sum_offset + i] = 0.0;

  rec->x1 = x + lb * dx;
  rec->y1 = y + lb * dy;
  rec->x2 = x + lf * dx;
  rec->y2 = y + lf * dy;
  rec->width = wl - wr;
  rec->x = x;
  rec->y = y;
  rec->theta = theta;
  rec->dx = dx;
  rec->dy = dy;
  rec->prec = prec;
  rec->p = p;
  rec->sum = sum;

  if( rec->width < 1.0 ) rec->width = 1.0;

  return elongation;
}

/*----------------------------------------------------------------------------*/
void V2::region_grow(Point2D<int> loc, Image<float>& angles, struct point * reg,
                         int * reg_size, float * reg_angle, Image<byte>& used,
                         float prec, int radius,
                         Image<float> modgrad, double p, int min_reg_size )
{
  float sumdx,sumdy;
  int xx,yy,i;

  /* first point of the region */
  *reg_size = 1;
  reg[0].x = loc.i;
  reg[0].y = loc.j;
  *reg_angle = angles.getVal(loc);
  sumdx = (float) cos( (double) (*reg_angle) );
  sumdy = (float) sin( (double) (*reg_angle) );
  used.setVal(loc, USED);

  /* try neighbors as new region points */
  //LINFO("Grow point %ix%i", loc.i, loc.j);
  //Image<PixRGB<byte> > tmp = modgrad;

  for(i=0; i<*reg_size; i++)
  {
    for(xx=reg[i].x-radius; xx<=reg[i].x+radius; xx++)
      for(yy=reg[i].y-radius; yy<=reg[i].y+radius; yy++)
      {
        if( xx>=0 && yy>=0 && xx<used.getWidth() && yy<used.getHeight() &&
            used[xx+yy*used.getWidth()] != USED)
        {
          //LINFO("Aligned %f %f = %i", *reg_angle, prec,
          //    isaligned(Point2D<int>(xx,yy),angles,*reg_angle,prec) );
          if ( isaligned(Point2D<int>(xx,yy),angles,*reg_angle,prec) )
          {
            /* add point */

            //tmp.setVal(xx,yy, PixRGB<byte>(255, 0,0));
            //SHOWIMG(tmp);

            used[xx+yy*used.getWidth()] = USED;
            reg[*reg_size].x = xx;
            reg[*reg_size].y = yy;
            ++(*reg_size);

            /* update reg_angle */
            sumdx += (float) cos( (double) angles[xx+yy*angles.getWidth()] );
            sumdy += (float) sin( (double) angles[xx+yy*angles.getWidth()] );
            *reg_angle = (float) atan2( (double) sumdy, (double) sumdx );
          }
        }
      }
  }
  //if (*reg_size > 5)
  //  SHOWIMG(tmp);

  if(itsLSDVerbose) /* print region points */
    {
      LINFO("region found: %d points\n",*reg_size);
      for(i=0; i<*reg_size; i++) fprintf(stderr,"| %d %d ",reg[i].x,reg[i].y);
      fprintf(stderr,"\n");
    }
}

/*----------------------------------------------------------------------------*/
double V2::rect_improve( struct rect * rec, Image<float>& angles,
                            double logNT, double eps )
{
  struct rect r;
  double nfa_val,nfa_new;
  float delta = 0.5;
  float delta_2 = delta / 2.0;
  int n;

  nfa_val = rect_nfa(rec,angles,logNT);

  if( nfa_val > eps ) return nfa_val;

  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      r.p /= 2.0;
      r.prec = M_PI * r.p;
      nfa_new = rect_nfa(&r,angles,logNT);
      if( nfa_new > nfa_val )
        {
          nfa_val = nfa_new;
          rect_copy(&r,rec);
        }
    }

  if( nfa_val > eps ) return nfa_val;

  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.width -= delta;
          nfa_new = rect_nfa(&r,angles,logNT);
          if( nfa_new > nfa_val )
            {
              rect_copy(&r,rec);
              nfa_val = nfa_new;
            }
        }
    }

  if( nfa_val > eps ) return nfa_val;

  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 += -r.dy * delta_2;
          r.y1 +=  r.dx * delta_2;
          r.x2 += -r.dy * delta_2;
          r.y2 +=  r.dx * delta_2;
          r.width -= delta;
          nfa_new = rect_nfa(&r,angles,logNT);
          if( nfa_new > nfa_val )
            {
              rect_copy(&r,rec);
              nfa_val = nfa_new;
            }
        }
    }

  if( nfa_val > eps ) return nfa_val;

  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 -= -r.dy * delta_2;
          r.y1 -=  r.dx * delta_2;
          r.x2 -= -r.dy * delta_2;
          r.y2 -=  r.dx * delta_2;
          r.width -= delta;
          nfa_new = rect_nfa(&r,angles,logNT);
          if( nfa_new > nfa_val )
            {
              rect_copy(&r,rec);
              nfa_val = nfa_new;
            }
        }
    }

  if( nfa_val > eps ) return nfa_val;

  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      r.p /= 2.0;
      r.prec = M_PI * r.p;
      nfa_new = rect_nfa(&r,angles,logNT);
      if( nfa_new > nfa_val )
        {
          nfa_val = nfa_new;
          rect_copy(&r,rec);
        }
    }

  return nfa_val;
}


/*----------------------------------------------------------------------------*/
/*-------------------------- LINE SEGMENT DETECTION --------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
std::vector<V2::LineSegment> V2::lineSegmentDetection(Image<float>& img, float q, float d, double eps,
                                  int n_bins, int max_grad, float scale, float sigma_scale )
{

  float rho = q / (float) sin( M_PI / (double) d );   /* gradient threshold */
  std::vector<LineSegment> lines;

  /* scale (if necesary) and angles computation */
  Image<float> in = img;
  //if( scale != 1.0 )
  //  in = gaussian_sampler( img, scale, sigma_scale );

  //struct coorlist * list_p;
  std::vector<Point2D<int> > list_p;
  void * mem_p;
  Image<float> modgrad;
  if (itsLSDVerbose) LINFO("Get Angles");
  Image<float> angles = ll_angle(in,rho,list_p,&mem_p,modgrad,n_bins,max_grad);


  int xsize = in.getWidth();
  int ysize = in.getHeight();

  double logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0;
  double p = 1.0 / (double) d;
  int min_reg_size = 5; //(int)(-logNT/log10(p)); /* minimal number of point that can
                          //                      give a meaningful event */

  Image<byte> used(xsize, ysize, ZEROS);
  struct point * reg = (struct point *) calloc(xsize * ysize, sizeof(struct point));

  int sum_res = 1;
  int sum_offset =(int)( sum_res * ceil( sqrt( (double) xsize * xsize + (double) ysize * ysize) ) + 2);
  float* sum_l = (float *) calloc(2*sum_offset,sizeof(float));
  float* sum_w = (float *) calloc(2*sum_offset,sizeof(float));
  if( !reg || !sum_l || !sum_w ) LFATAL("Not enough memory!\n");
  for(int i=0;i<2*sum_offset;i++) sum_l[i] = sum_w[i] = 0;

  /* just start at point x,y option */
  //if( x && y && *x > 0 && *y > 0 && *x < (xsize-1) && *y < (ysize-1) )
  //  {
  //    if(itsLSDVerbose) fprintf(stderr,"starting only at point %d,%d.\n",*x,*y);
  //    list_p = (struct coorlist *) mem_p;
  //    list_p->x = *x;
  //    list_p->y = *y;
  //    list_p->next = NULL;
  //  }

  if (itsLSDVerbose) LINFO("Search for line segments");
  /* search for line segments */
  int segnum = 0;
  for(uint pIdx = 0; pIdx < list_p.size(); pIdx++)
    if( ( used[ list_p[pIdx].i + list_p[pIdx].j * used.getWidth() ] == NOTUSED &&
          angles[ list_p[pIdx].i + list_p[pIdx].j * angles.getWidth() ] != NOTDEF)  )
      {
          if (itsLSDVerbose) LINFO("try to find a line segment starting on %d,%d.\n",
                  list_p[pIdx].i,list_p[pIdx].j);

          /* find the region of connected point and ~equal angle */
          int reg_size;
          float reg_angle;
          float prec = M_PI / d;
          int radius = 1;
          region_grow( list_p[pIdx], angles, reg, &reg_size,
              &reg_angle, used, prec, radius,
              modgrad, p, min_reg_size );

        /* just process regions with a minimum number of points */
        if( reg_size < min_reg_size )
          {
           // LINFO("Region too small %i %i", reg_size, min_reg_size);
            if(itsLSDVerbose) LINFO("region too small, discarded.\n");
            for(int i=0; i<reg_size; i++)
              used[reg[i].x+reg[i].y*used.getWidth()] = NOTINI;
            continue;
          }

        if(itsLSDVerbose) LINFO("rectangular approximation of region.\n");
        struct rect rec;
        region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&rec, sum_l, sum_w, sum_offset, sum_res);

        if(itsLSDVerbose) LINFO("improve rectangular approximation.\n");
        double nfa_val = rect_improve(&rec,angles,logNT,eps);

        LINFO("nfa_val %f", nfa_val);
        if( nfa_val > eps )
        {
          if(itsLSDVerbose) LINFO("line segment found! num %d nfa %g\n",
              segnum,nfa_val);

          /*
             0.5 must be added to compensate that the gradient was
             computed using a 2x2 window, so the computed gradient
             is valid at a point with offset (0.5,0.5). The coordinates
             origin is at the center of pixel 0,0.
             */
          rec.x1 += 0.5; rec.y1 += 0.5;
          rec.x2 += 0.5; rec.y2 += 0.5;

          if( scale != 1.0 )
          {
            rec.x1 /= scale;
            rec.y1 /= scale;
            rec.x2 /= scale;
            rec.y2 /= scale;
            rec.width /= scale;
          }

          LineSegment ls;
          ls.p1.i = rec.x1;
          ls.p1.j = rec.y1;
          ls.p2.i = rec.x2;
          ls.p2.j = rec.y2;
          ls.width = rec.width;
          ls.length = sqrt( squareOf(rec.x1 - rec.x2) + squareOf(rec.y1 - rec.y2));
          ls.center.i = rec.x;
          ls.center.j = rec.y;
          ls.ori = atan2(rec.y1-rec.y2,rec.x2-rec.x1);
          ls.strength = 1;

          //Find the color on etiher size of the line
          //float len = 7;
          //int x1 = int(cos(ls.ori-(M_PI/2))*len/2);
          //int y1 = int(sin(ls.ori-(M_PI/2))*len/2);

          //Point2D<int> p1((int)ls.center.i-x1, (int)ls.center.j+y1);
          //Point2D<int> p2((int)ls.center.i+x1, (int)ls.center.j-y1);
          //if (img.coordsOk(p1))
          //{
          //  ls.side1Color.clear();
          //  ls.side1Color.push_back(itsLGNInput[0].getVal(p1));
          //  ls.side1Color.push_back(itsLGNInput[1].getVal(p1));
          //  ls.side1Color.push_back(itsLGNInput[2].getVal(p1));
          //}

          //if (img.coordsOk(p2))
          //{
          //  ls.side2Color.clear();
          //  ls.side2Color.push_back(itsLGNInput[0].getVal(p2));
          //  ls.side2Color.push_back(itsLGNInput[1].getVal(p2));
          //  ls.side2Color.push_back(itsLGNInput[2].getVal(p2));
          //}

          lines.push_back(ls);


          //Show regions

          //fprintf(arout,"region %d:",segnum);
          //for(i=0; i<reg_size; i++)
          //  fprintf(arout," %d,%d;",reg[i].x,reg[i].y);
          //fprintf(arout,"\n");

          ++segnum;
        }
        else
          for(int i=0; i<reg_size; i++)
            used[reg[i].x+reg[i].y*used.getWidth()] = NOTINI;
      }


//  getchar();

  return lines;
}

std::vector<V2::LineSegment> V2::lineSegmentDetection(const TensorField& tensorField,
    float q, float d, double eps,
    int n_bins, int max_grad )
{
  //float rho = q / (float) sin( M_PI / (double) d );   /* gradient threshold */
  std::vector<LineSegment> lines;

  //struct coorlist * list_p;
  std::vector<Point2D<int> > list_p;
  Image<float> modgrad;
  if (itsLSDVerbose) LINFO("Get Angles");
  Image<float> angles = ll_angle(tensorField,max_grad*0.10,list_p,modgrad,n_bins,max_grad);

  int xsize = tensorField.t1.getWidth();
  int ysize = tensorField.t1.getHeight();

  double logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0;
  double p = 1.0 / (double) d;
  int min_reg_size = 5; //(int)(-logNT/log10(p)); /* minimal number of point that can
                         //                       give a meaningful event */

  Image<byte> used(xsize, ysize, ZEROS);
  struct point * reg = (struct point *) calloc(xsize * ysize, sizeof(struct point));

  int sum_res = 1;
  int sum_offset =(int)( sum_res * ceil( sqrt( (double) xsize * xsize + (double) ysize * ysize) ) + 2);
  float* sum_l = (float *) calloc(2*sum_offset,sizeof(float));
  float* sum_w = (float *) calloc(2*sum_offset,sizeof(float));
  if( !reg || !sum_l || !sum_w ) LFATAL("Not enough memory!\n");
  for(int i=0;i<2*sum_offset;i++) sum_l[i] = sum_w[i] = 0;

  if (itsLSDVerbose) LINFO("Search for line segments");
  /* search for line segments */
  int segnum = 0;
  for(uint pIdx = 0; pIdx < list_p.size(); pIdx++)
    if( ( used[ list_p[pIdx].i + list_p[pIdx].j * used.getWidth() ] == NOTUSED &&
          angles[ list_p[pIdx].i + list_p[pIdx].j * angles.getWidth() ] != NOTDEF)  )
      {
          if (itsLSDVerbose) LINFO("try to find a line segment starting on %d,%d.\n",
                  list_p[pIdx].i,list_p[pIdx].j);

          /* find the region of connected point and ~equal angle */
          int reg_size;
          float reg_angle;
          float prec = M_PI / d;
          int radius = 2;
          region_grow( list_p[pIdx], angles, reg, &reg_size,
              &reg_angle, used,
              prec, radius,
              modgrad, p, min_reg_size );

        /* just process regions with a minimum number of points */
        if( reg_size < min_reg_size )
          {
            if(itsLSDVerbose) LINFO("region too small, discarded.\n");
            for(int i=0; i<reg_size; i++)
              used[reg[i].x+reg[i].y*used.getWidth()] = NOTINI;
            continue;
          }

        if(itsLSDVerbose) LINFO("rectangular approximation of region.\n");
        struct rect rec;
        region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&rec, sum_l, sum_w, sum_offset, sum_res);

        if(itsLSDVerbose) LINFO("improve rectangular approximation.\n");
        double nfa_val = rect_improve(&rec,angles,logNT,eps);

        if ( fabs(nfa_val) > eps ) 
        {
          if(itsLSDVerbose) LINFO("line segment found! num %d nfa %g\n",
              segnum,nfa_val);

          /*
             0.5 must be added to compensate that the gradient was
             computed using a 2x2 window, so the computed gradient
             is valid at a point with offset (0.5,0.5). The coordinates
             origin is at the center of pixel 0,0.
             */
          rec.x1 += 0.5; rec.y1 += 0.5;
          rec.x2 += 0.5; rec.y2 += 0.5;

          if (rec.x1 >= xsize) rec.x1 = xsize-1;
          if (rec.y1 >= ysize) rec.y1 = ysize-1;

          if (rec.x2 >= xsize) rec.x2 = xsize-1;
          if (rec.y2 >= ysize) rec.y2 = ysize-1;

          LineSegment ls;
          ls.p1.i = rec.x1;
          ls.p1.j = rec.y1;
          ls.p2.i = rec.x2;
          ls.p2.j = rec.y2;
          ls.width = rec.width;
          ls.length = sqrt( squareOf(rec.x1 - rec.x2) + squareOf(rec.y1 - rec.y2));
          ls.center.i = rec.x;
          ls.center.j = rec.y;
          ls.ori = atan2(rec.y1-rec.y2,rec.x2-rec.x1);
          ls.prob = rec.sum;

          if (ls.ori < 0) ls.ori += M_PI;
          if (ls.ori >= M_PI) ls.ori -= M_PI;

          ls.strength = 1;

          //Find the color on etiher size of the line
          //float len = 7;
          //int x1 = int(cos(ls.ori-(M_PI/2))*len/2);
          //int y1 = int(sin(ls.ori-(M_PI/2))*len/2);

          //Point2D<int> p1((int)ls.center.i-x1, (int)ls.center.j+y1);
          //Point2D<int> p2((int)ls.center.i+x1, (int)ls.center.j-y1);
          //if (img.coordsOk(p1))
          //{
          //  ls.side1Color.clear();
          //  ls.side1Color.push_back(itsLGNInput[0].getVal(p1));
          //  ls.side1Color.push_back(itsLGNInput[1].getVal(p1));
          //  ls.side1Color.push_back(itsLGNInput[2].getVal(p1));
          //}

          //if (img.coordsOk(p2))
          //{
          //  ls.side2Color.clear();
          //  ls.side2Color.push_back(itsLGNInput[0].getVal(p2));
          //  ls.side2Color.push_back(itsLGNInput[1].getVal(p2));
          //  ls.side2Color.push_back(itsLGNInput[2].getVal(p2));
          //}

          lines.push_back(ls);


          //Show regions

          //fprintf(arout,"region %d:",segnum);
          //for(i=0; i<reg_size; i++)
          //  fprintf(arout," %d,%d;",reg[i].x,reg[i].y);
          //fprintf(arout,"\n");

          ++segnum;
        }
        else
          for(int i=0; i<reg_size; i++)
            used[reg[i].x+reg[i].y*used.getWidth()] = NOTINI;
      }


//  getchar();

  free(reg);
  free(sum_l);
  free(sum_w);

  return lines;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

