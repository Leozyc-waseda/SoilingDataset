/*!@file AppPsycho/test-parseScanpath.C Parses the eye scanpath and recognizes
  each fixation */

// ////////////////////////////////////////////////////////////////////
// // The iLab Neuromorphic Vision C++ Toolkit - Copyright (C)
// 2000-2002 // by the University of Southern California (USC)
// and the iLab at USC.  // See http://iLab.usc.edu for
// information about this project.  //
// ////////////////////////////////////////////////////////////////////
// // Major portions of the iLab Neuromorphic Vision Toolkit
// are protected // under the U.S. patent ``Computation of
// Intrinsic Perceptual Saliency // in Visual Environments,
// and Applications'' by Christof Koch and // Laurent Itti,
// California Institute of Technology, 2001 (patent //
// pending; application number 09/912,225 filed July 23, 2001;
// see // http://pair.uspto.gov/cgi-bin/final/home.pl for
// current status).  //
// ////////////////////////////////////////////////////////////////////
// // This file is part of the iLab Neuromorphic Vision C++
// Toolkit.  // // The iLab Neuromorphic Vision C++ Toolkit is
// free software; you can // redistribute it and/or modify it
// under the terms of the GNU General // Public License as
// published by the Free Software Foundation; either //
// version 2 of the License, or (at your option) any later
// version.  // // The iLab Neuromorphic Vision C++ Toolkit is
// distributed in the hope // that it will be useful, but
// WITHOUT ANY WARRANTY; without even the // implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR // PURPOSE.
// See the GNU General Public License for more details.  // //
// You should have received a copy of the GNU General Public
// License // along with the iLab Neuromorphic Vision C++
// Toolkit; if not, write // to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, // Boston, MA 02111-1307
// USA.  //
// ////////////////////////////////////////////////////////////////////
// //
//
// Primary maintainer for this file: Vidhya Navalpakkam <navalpak@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/test-parseScanpath.C $
// $Id: test-parseScanpath.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Component/ModelManager.H"
#include "Image/Pixels.H"
#include "Image/Point2D.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("parse scanpath");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<input.ppm> <rows> <columns> <coordsFile>"
                               " <fixnsFile> <reco> <traj.ppm>"
                               " <saccadeEndPoints>"
                               " <target> <less> <same> <more>"
                               " <'d' for draw / 'r' for recognize /"
                               " 'g' for generate stimuli>",
                               1, -1) == false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // ------------------ READ INPUT --------------------------
  // get the arguments
  int rows = manager.getExtraArgAs<int>(1);
  int cols = manager.getExtraArgAs<int>(2);
  const char* coords = manager.getExtraArg (3).c_str();
  const char* fixns = manager.getExtraArg (4).c_str();
  const char* reco = manager.getExtraArg (5).c_str();
  const char* output = manager.getExtraArg (6).c_str();
  const char* saccTargets = manager.getExtraArg (7).c_str();
  const char* targetName = manager.getExtraArg (8).c_str();
  const char* lessName = manager.getExtraArg (9).c_str();
  const char* sameName = manager.getExtraArg (10).c_str();
  const char* moreName = manager.getExtraArg (11).c_str();
  const char* option = manager.getExtraArg (12).c_str();

  // ------------------ INITIALIZE PARAMS -------------------------
  if (option[0] == 'd')
    {
      Image<PixRGB< byte> > traj =
        Raster::ReadRGB(manager.getExtraArg (0));
      // draw the scanpath
      LINFO (" image grid: %d x %d", cols, rows);
      int w = traj.getWidth();
      int h = traj.getHeight();
      // box size in the grid
      int bw = w / cols;
      int bh = h / rows;
      // draw the grid
          // draw the horizontal lines
      for (int i = 1; i < rows; i++)
        {
          Point2D<int> start (0, i * bh), end (w, i * bh);
          drawLine (traj, start, end, PixRGB<byte>(128, 128, 128), 1);
        }
      // draw the vertical lines
      for (int i = 1; i < cols; i++)
        {
          Point2D<int> start (i * bw, 0), end (i * bw, h);
          drawLine (traj, start, end, PixRGB<byte>(128, 128, 128), 1);
        }

      // draw the scanpath
      std::ifstream fil (coords);
      if (fil == 0) LFATAL ("Couldn't open the coords file: %s", coords);
      // read all the coords
      std::string line;
      std::vector< std::string > xy;
      while (getline (fil, line))
        {
          int x_idx = line.find (" ");
          std::string x = line.substr (0, x_idx);
          line = line.substr (x_idx + 1);
          int y_idx = line.find (" ");
          std::string y = line.substr (0,  y_idx - 1);
          xy.push_back (x);
          xy.push_back (y);
        }

      // set some loop params
      Point2D<int> prev_fixn (-1, -1);
      for (uint i = 0; i < xy.size(); i = i + 2)
        {
          Point2D<int> fixn (atoi (xy[i].c_str()), atoi (xy[i+1].c_str()));
          drawCircle(traj, fixn, 2, PixRGB<byte>(255, 0, 0) , 2);
          if (i != 0)
            drawLine(traj, prev_fixn, fixn, PixRGB<byte>(255, 255, 0), 1);
          prev_fixn = fixn;
        }

      // draw fixations
      std::ifstream fix (fixns);
      if (fix == 0) LFATAL ("Couldn't open the fixations file: %s", fixns);
      xy.clear();
      while (getline (fix, line))
        {
          int x_idx = line.find (" ");
          std::string x = line.substr (0, x_idx);
          line = line.substr (x_idx + 1);
          int y_idx = line.find (" ");
          std::string y = line.substr (0,  y_idx - 1);
          xy.push_back (x);
          xy.push_back (y);
        }
      // draw saccade targets
      std::ifstream sac (saccTargets);
      if (sac == 0) LFATAL ("Couldn't open the saccade targets file: %s",
                            saccTargets);
      xy.clear();
      while (getline (sac, line))
        {
          int x_idx = line.find (" ");
          std::string x = line.substr (0, x_idx);
          line = line.substr (x_idx + 1);
          int y_idx = line.find (" ");
          std::string y = line.substr (0,  y_idx - 1);
          xy.push_back (x);
          xy.push_back (y);
        }
      for (uint i = 0; i < xy.size(); i = i + 2)
        {
          Point2D<int> sacEndPt (atoi (xy[i].c_str()), atoi (xy[i+1].c_str()));
          drawCircle(traj, sacEndPt, 2, PixRGB<byte>(0, 255, 0) , 4);
        }
      Raster::WriteRGB (traj, output, RASFMT_PNM);
    }
  //***************** GENERATE STIMULI AND DRAW SCANPATHS *******************
  if (option[0] == 'g')
    {
      Image<PixRGB< byte> > traj =
        Raster::ReadRGB(manager.getExtraArg (0));
      // draw the scanpath
      LINFO (" image grid: %d x %d", cols, rows);
      int w = traj.getWidth();
      int h = traj.getHeight();
      // box size in the grid
      int bw = w / cols;
      int bh = h / rows;
      // draw the grid
          // draw the horizontal lines
      for (int i = 1; i < rows; i++)
        {
          Point2D<int> start (0, i * bh), end (w, i * bh);
          drawLine (traj, start, end, PixRGB<byte>(128, 128, 128), 1);
        }
      // draw the vertical lines
      for (int i = 1; i < cols; i++)
        {
          Point2D<int> start (i * bw, 0), end (i * bw, h);
          drawLine (traj, start, end, PixRGB<byte>(128, 128, 128), 1);
        }
      // mark each cell in the grid with less/mid/high
      std::ifstream rec (reco);
      if (rec == 0) LFATAL  ("Couldn't open the reco file: %s", reco);
      std::string line;
      while (getline (rec, line))
        {
          char msg[2];
          int idx = line.find_last_of (" ");
          std::string itemName = line.substr (idx + 1);
          line = line.substr (0, idx);
          idx = line.find_last_of (" ");
          int y = atoi (line.substr (idx + 1).c_str());
          line = line.substr (0, idx);
          idx = line.find_last_of (" ");
          int x = atoi (line.substr (idx + 1).c_str());
          if (itemName.find(targetName,0) != std::string::npos)
            sprintf (msg, "T");
          if (itemName.find(lessName,0) != std::string::npos)
            sprintf (msg, "L");
          if (itemName.find(sameName,0) != std::string::npos)
            sprintf (msg, "M");
          if (itemName.find(moreName,0) != std::string::npos)
            sprintf (msg, "H");
          writeText (traj, Point2D<int>(x,y), msg,PixRGB<byte>(255),PixRGB<byte>(0));
        }
      // draw the scanpath
      std::ifstream fil (coords);
      if (fil == 0) LFATAL ("Couldn't open the coords file: %s", coords);
      // read all the coords
      std::vector< std::string > xy;
      while (getline (fil, line))
        {
          int x_idx = line.find (" ");
          std::string x = line.substr (0, x_idx);
          line = line.substr (x_idx + 1);
          int y_idx = line.find (" ");
          std::string y = line.substr (0,  y_idx - 1);
          xy.push_back (x);
          xy.push_back (y);
        }

      // set some loop params
      Point2D<int> prev_fixn (-1, -1);
      for (uint i = 0; i < xy.size(); i = i + 2)
        {
          Point2D<int> fixn (atoi (xy[i].c_str()), atoi (xy[i+1].c_str()));
          drawCircle(traj, fixn, 2, PixRGB<byte>(255, 0, 0) , 2);
          if (i != 0)
            drawLine(traj, prev_fixn, fixn, PixRGB<byte>(255, 255, 0), 1);
          prev_fixn = fixn;
        }

      // draw fixations
      std::ifstream fix (fixns);
      if (fix == 0) LFATAL ("Couldn't open the fixations file: %s", fixns);
      xy.clear();
      while (getline (fix, line))
        {
          int x_idx = line.find (" ");
          std::string x = line.substr (0, x_idx);
          line = line.substr (x_idx + 1);
          int y_idx = line.find (" ");
          std::string y = line.substr (0,  y_idx - 1);
          xy.push_back (x);
          xy.push_back (y);
        }
      /*
      for (uint i = 0; i < xy.size(); i = i + 2)
        {
          Point2D<int> fixn (atoi (xy[i].c_str()), atoi (xy[i+1].c_str()));
          drawCircle(traj, fixn, 2, PixRGB<byte>(0, 0, 255) , 4);
        }
      */
      // draw saccade targets
      std::ifstream sac (saccTargets);
      if (sac == 0) LFATAL ("Couldn't open the saccade targets file: %s",
                            saccTargets);
      xy.clear();
      while (getline (sac, line))
        {
          int x_idx = line.find (" ");
          std::string x = line.substr (0, x_idx);
          line = line.substr (x_idx + 1);
          int y_idx = line.find (" ");
          std::string y = line.substr (0,  y_idx - 1);
          xy.push_back (x);
          xy.push_back (y);
        }
      for (uint i = 0; i < xy.size(); i = i + 2)
        {
          Point2D<int> sacEndPt (atoi (xy[i].c_str()), atoi (xy[i+1].c_str()));
          drawCircle(traj, sacEndPt, 2, PixRGB<byte>(0, 255, 0) , 4);
        }
      Raster::WriteRGB (traj, output, RASFMT_PNM);
    }

  //***************** RECOGNIZE SACCADIC ENDPOINTS *********************
  else if (option[0] == 'r')
    {
      // recognize the fixations
      std::ifstream rec (reco);
      if (rec == 0) LFATAL  ("Couldn't open the reco file: %s", reco);
      /* 6 distractors: s2, s3, m2, m3, n2, n3
         4 item sper distractor
         x, y per item
      */
      int d[3][8][2], t[2];
      t[0] = 0; t[1] = 0; // avoid compiler warning
      //initialize the distances
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 8; j++){
          d[i][j][0] = 1024;d[i][j][1]=1024;
        }
      std::string line;
      int num[3];
      for (int i = 0; i < 3; i ++)
        num[i] = 0; // initialize

       while (getline (rec, line))
        {
          int idx = line.find_last_of (" ");
          std::string itemName = line.substr (idx + 1);
          line = line.substr (0, idx);
          idx = line.find_last_of (" ");
          int y = atoi (line.substr (idx + 1).c_str());
          line = line.substr (0, idx);
          idx = line.find_last_of (" ");
          int x = atoi (line.substr (idx + 1).c_str());
          //LINFO ("item %s at (%d, %d)", itemName.c_str(), x, y);
          if (itemName.find(targetName,0) != std::string::npos)
            {
              t[0] = x;
              t[1] = y;
            }
          if (itemName.find(lessName,0) != std::string::npos)
            {
              d[0][num[0]][0] = x;
              d[0][num[0]][1] = y;
              num[0] += 1;
            }
          if (itemName.find(sameName,0) != std::string::npos)
            {
              d[1][num[1]][0] = x;
              d[1][num[1]][1] = y;
              num[1] += 1;
            }
          if (itemName.find(moreName,0) != std::string::npos)
            {
              d[2][num[2]][0] = x;
              d[2][num[2]][1] = y;
              num[2] += 1;
            }
        }
       int score[4], count[4]; float nearest[4];
       // 3 types of dtarctors, 1 target
       for (int i = 0; i < 4; i ++){
         score[i] = 0; count[i] = 0; nearest[i] = 1024.0f;
       }
       std::ifstream sac (saccTargets);
       if (sac == 0) LFATAL ("Couldn't open the saccade targets file: %s",
                             saccTargets);
       // read all the saccade targets
       int numFix = 1;
       // distance to the nearest category
       float minD = 1024.0f;

       while (getline (sac, line))
         {
           // recognize the current fixn
           // score: 1 for the nearest item, 0 for others
           int idx = line.find_last_of (" ");
           int y = atoi (line.substr (idx + 1).c_str());
           int x = atoi (line.substr (0, idx).c_str());
           // find distance btw fixn and distractors
           for (int i = 0; i < 3; i ++)
             for (int j = 0; j < 8; j ++)
               {
                 float dist = sqrt ((d[i][j][0] - x)*(d[i][j][0] - x) +
                                    (d[i][j][1] - y)*(d[i][j][1] - y));
                 if (dist < nearest[i])
                   nearest[i] = dist;
               }
          // distance btw target and fixn
          nearest[3] = sqrt ((t[0] - x)*(t[0] - x) + (t[1] - y)*(t[1] - y));

          // find the nearest category
          for (int i = 0; i < 4; i ++)
            if (nearest[i] < minD)
              minD = nearest[i];

          // check if there are multiple nearest items
          for (int i = 0; i < 4; i ++){
            if (nearest[i] == minD) count[i] ++;
            // assign the fixation to the nearest category
            score[i] += count[i];
          }
          // display the results of the current fixn.
          //LINFO ("fix%d: %f %f %f %f", numFix, nearest[0],
          //     nearest[1], nearest[2],nearest[3]);
          LINFO ("fix%d: %d %d %d %d", numFix, count[0],
                 count[1], count[2], count[3]);

          numFix ++;
          // reset count values
          for (int i  = 0; i < 4; i ++){
            count[i] = 0; nearest[i] = 1024.0f;
          }
          // reset distance to the nearest category
          minD = 1024.0f;
         }
       // display cumulative results
       LINFO ("total: %d %d %d %d", score[0], score[1],
              score[2], score[3]);

    }
  manager.stop();
  manager.saveConfig();

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
