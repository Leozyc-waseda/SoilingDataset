/*!@file AppPsycho/searcharray2d.C create a randomized search array from
  image patches of a single target and the rest for distractors*/

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/searcharray2d.C $
// $Id: searcharray2d.C 14376 2011-01-11 02:44:34Z pez $
//

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "Util/Assert.H"
#include "Util/MathFunctions.H"
#include "Util/StringConversions.H"
#include "Util/log.H"

#include <cstdio>
#include <math.h>
#include <unistd.h>
#include <vector>

void image_patch(const Image< PixRGB<byte> >& patch, const int ti,
                 const int tj, Image< PixRGB<byte> >& image,
                 const double alpha, const float FACTORx,
                 const float FACTORy,
                 Image<byte>& targets,
                 bool do_target, Image< PixRGB<byte> >& randomGrid,
                 const Image< PixRGB<byte> >& randomNum);


/*! This program generates a randomized search array (and associated
  target mask) from several image patches (1 for target and the rest
  for distractors). */

int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("search array");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<target.ppm> <w> <alpha> <noise>"
                               "<numRows> <numCols> <output>"
                               "<reco> <noCheat> <checkResponse> <pathNumbers>"
                               "<distractor_1.ppm> ... <distractor_n.ppm>",
                               1, -1)==false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // ------------------ READ INPUT --------------------------
  // get the arguments
  Image<PixRGB<byte> > target =
    Raster::ReadRGB(manager.getExtraArg (0));
  int w = manager.getExtraArgAs<int>(1);
  float alpha = manager.getExtraArgAs<float>(2);
  float noise = manager.getExtraArgAs<float>(3);
  int rows = manager.getExtraArgAs<int>(4);
  int cols = manager.getExtraArgAs<int>(5);
  const char* output = manager.getExtraArg (6).c_str();
  const char* reco = manager.getExtraArg (7).c_str();
  const char* noCheat = manager.getExtraArg (8).c_str();
  const char* checkResponse = manager.getExtraArg (9).c_str();
  std::string pathNumbers = manager.getExtraArg (10);
  uint numD = manager.numExtraArgs() - 11;
  Image<PixRGB<byte> >* distractor[numD];
  for (uint i = 0; i < numD; i++)
    {
      // read the distractor patches
      distractor[i] = new Image<PixRGB<byte> >
        (Raster::ReadRGB(manager.getExtraArg (11 + i)));
    }
  int pw = target.getWidth(), ph = target.getHeight();

  // ------------------ INITIALIZE PARAMS -------------------------
  for (uint i = 0; i < numD; i++)
    {
      //all targets equal size
      ASSERT(pw == distractor[i]->getWidth() &&
             ph == distractor[i]->getHeight());
    }

  initRandomNumbers();

  // initialize results:
  Image<PixRGB<byte> > image(w, w, ZEROS);
  Image<PixRGB<byte> > randomGrid(w, w, ZEROS);
  Image<byte> targets(w, w, ZEROS);

  float FACTORx = w / (cols * pw);
  float FACTORy = w / (rows * ph);
  int numBox = cols * rows;
  int numPerD = int((numBox - 1) / numD);
  LINFO (" image grid: %d x %d, number per distractor type = %d",
         cols, rows, numPerD);

  // read the 2 digit randomNumber patches: to avoid any head
  // movements while entering the digits, we restrict each digit to
  // 1..5, hence a max of 25 numbers
  /*int numR, numRows, numCols;
  if (numBox > 25)
    {
      numRows = 5;
      numCols = 5;
      numR = 25;
    }
  else
    {
      numRows = rows;
      numCols = cols;
      numR = numBox;
    }*/
  Image<PixRGB<byte> >* numbers[25];
  std::vector <std::string> names;

  // let's display some numbers to be displayed at each locn
  for (int i = 0; i < 5; i ++)
    for (int j = 0; j < 5; j++)
      {
        int k = i * 5 + j;
        names.push_back (pathNumbers + "/" + toStr (i+1) + toStr (j+1) +
                         ".ppm");
        numbers[k] = new Image<PixRGB<byte> >
          (Raster::ReadRGB(names[k]));
        //all targets equal size
        ASSERT(pw == numbers[k]->getWidth() &&
               ph == numbers[k]->getHeight());
      }


  // --------------- MAKE THE SEARCH ARRAY AND RANDOM GRID --------------
  // keep a list of all locns in the grid where items can appear
  int locn[numBox][2];
  int box[numBox]; // dummy for randomizing

  FILE* f = fopen (reco, "w");
  FILE* check = fopen (checkResponse, "w");
  for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
      {
        int k = i * cols + j;
        box[k] = k; // intialize with index
        // the above box is at locn. <i+1, j+1> in the grid
        locn[k][0] = i + 1;
        locn[k][1] = j + 1;
      }
   // list of indices that will be randomized for flashing numbers
  int random[25];
  for (int i = 0; i < 5; i ++)
    for (int j = 0; j < 5; j++){
      int k = i * 5 + j;
      random[k] = k; // intialize now: randomize later
    }

  // shuffle the grid to randomize the locn of the target and
  // distractors
  randShuffle (box, numBox);

  // also randomize the display of numbers in the grid
  randShuffle (random, 25);

  // allocate the target to the randomized box
  int idx = 0; // index for the box and random array
  int r = locn[box[idx]][0];
  int c = locn[box[idx]][1];
  // fprintf (f, "%d %d %s\n", r, c, manager.getExtraArg (0).c_str());
  LINFO("-- TARGET AT (%d, %d)", r, c);
  image_patch (target, c-1, r-1, image, alpha, FACTORx, FACTORy, targets, 1,
               randomGrid, *numbers[random[idx]]);

  // record the random number at the target locn to compare response
  int start = names[random[idx]].rfind ("/");
  int end = names[random[idx]].rfind (".ppm");
  fprintf (check, "%s", names[random[idx]].
           substr (start+1, end-start-1).c_str());
  fclose (check);
  // record the  random number displayed at this locn in the reco file
  fprintf (f, "%d %d %s %s\n", r, c,
                 manager.getExtraArg(0).c_str(),
                 names[random[idx]].substr (start+1, end-start-1).c_str());
  idx++;

  // allocate the items to the randomized boxes
  for (uint disType = 0; disType < numD; disType ++)
    for (int count = 0; count < numPerD; count ++)
      {
        r = locn[box[idx]][0];
        c = locn[box[idx]][1];
        // record the random number displayed at this locn
        int start = names[random[idx]].rfind ("/");
        int end = names[random[idx]].rfind (".ppm");
        fprintf (f, "%d %d %s %s\n", r, c,
                 manager.getExtraArg(11 + disType).c_str(),
                 names[random[idx]].substr (start+1, end-start-1).c_str());
        LINFO("-- distractor %d AT (%d, %d)", disType + 1, r, c);
        image_patch (*distractor[disType], c-1, r-1, image, alpha,
                     FACTORx, FACTORy,
                     targets, 0, randomGrid, *numbers[random[idx]]);
        idx++;
      }

  // ------------------- ADD NOISE -----------------------------------
  /* add noise */
  Image< PixRGB<byte> >::iterator iptr = image.beginw(),
    stop = image.endw();
  while(iptr != stop)
    {
      if (randomDouble() <= noise)
        {
          if (randomDouble() >= 0.5) iptr->setRed(255);
          else iptr->setRed(0);
          if (randomDouble() >= 0.5) iptr->setGreen(255);
          else iptr->setGreen(0);
          if (randomDouble() >= 0.5) iptr->setBlue(255);
          else iptr->setBlue(0);
        }
      iptr ++;
    }

  // ----------------------- WRITE OUTPUT -------------------------------
  fclose (f);
  Raster::WriteRGB(image,  output, RASFMT_PNM);
  Raster::WriteRGB(randomGrid,  noCheat, RASFMT_PNM);
  Raster::WriteGray(targets,  output, RASFMT_PNM);

  // erase the memory used
  for (uint i = 0; i < numD; i++)
    {
      delete (distractor[i]);
    }
  for (int i = 0; i < 25; i++)
    {
      delete (numbers[i]);
    }
  manager.stop();
  return 0;
}

// ######################################################################
void image_patch(const Image< PixRGB<byte> >& patch, const int ti,
                 const int tj, Image< PixRGB<byte> >& image,
                 const double alpha,  const float FACTORx,
                 const float FACTORy, Image<byte>& targets,
                 bool do_target, Image< PixRGB<byte> >& randomGrid,
                 const Image< PixRGB<byte> >& randomNum)
{
  int pw = patch.getWidth(), ph = patch.getHeight();
  int w = image.getWidth();

  //int jitx = int(randomDouble() * (FACTOR - 1.0) * pw);
  //int jity = int(randomDouble() * (FACTOR - 1.0) * ph);
  int jitx = int(randomDouble() * (FACTORx - 2.0) * pw);
  int jity = int(randomDouble() * (FACTORy - 2.0) * ph);

  float jita = float(alpha * 3.14159 / 180.0 * (randomDouble() - 0.5) * 2.0);
  int offset = int(w - floor(w / (pw * FACTORx)) * (pw * FACTORx)) / 2;

  PixRGB<byte> zero(0, 0, 0);
  int px = 0, py = 0;

  for (double y = int(tj *ph * FACTORy); y < int(tj * ph * FACTORy + ph); y ++)
    {
      for (double x = int(ti * pw * FACTORx); x < int(ti * pw * FACTORx + pw);
           x ++)
        {
          int x2 = int(x + jitx + offset);
          int y2 = int(y + jity + offset);

          /* Shifting back and forth the center of rotation.*/
          double px2 = px - pw / 2.0F;
          double py2 = py - ph / 2.0F;

          float px3 = float(cos(jita) * px2 + sin(jita) * py2 + pw / 2.0F);
          float py3 = float(-sin(jita) * px2 + cos(jita) * py2 + pw / 2.0F);

          if (px3 < 0 || px3 >= pw || py3 < 0 || py3 >= ph )
            {
              image.setVal(x2, y2, zero);
              randomGrid.setVal(x2, y2, zero);
            }
          else
            {
              image.setVal(x2, y2, patch.getValInterp(px3, py3));
              randomGrid.setVal(x2, y2, randomNum.getVal((int)px3, (int)py3));
              if (do_target)
                {
                  if (patch.getVal(int(px3), int(py3)) == zero)
                    targets.setVal(x2, y2, 0);
                  else
                    targets.setVal(x2, y2, 255);
                }
            }
          px ++;
        }
      py ++;
      px = 0;
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
