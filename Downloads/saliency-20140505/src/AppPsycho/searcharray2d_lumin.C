/*!@file AppPsycho/searcharray2d_lumin.C Creates a randomized search array from
  image patches of a single target and many distractors occuring in
  randomized luminance*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/searcharray2d_lumin.C $
// $Id: searcharray2d_lumin.C 12074 2009-11-24 07:51:51Z itti $
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
#include <vector>

void image_patch(const Image< PixRGB<byte> >& patch, const int ti,
                 const int tj, Image< PixRGB<byte> >& image,
                 const double alpha, const float FACTOR,
                 Image<byte>& targets,
                 bool do_target, Image< PixRGB<byte> >& randomGrid,
                 const Image< PixRGB<byte> >& randomNum,
                 int& center_x, int& center_y);


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
                               "<target> <w> <alpha> <noise> <space>"
                               " <output>"
                               "<reco> <noCheat> <checkResponse> <pathNumbers>"
                               "<distractor_1> ... <distractor_3>"
                               "<lumin_1> ... <lumin_k>",
                               1, -1)==false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // ------------------ READ INPUT --------------------------
  // get the arguments
  const char* targetName = manager.getExtraArg (0).c_str();
  int w = manager.getExtraArgAs<int>(1);
  float alpha = manager.getExtraArgAs<float>(2);
  float noise = manager.getExtraArgAs<float>(3);
  float FACTOR = manager.getExtraArgAs<float>(4);
  const char* output = manager.getExtraArg (5).c_str();
  const char* reco = manager.getExtraArg (6).c_str();
  const char* noCheat = manager.getExtraArg (7).c_str();
  const char* checkResponse = manager.getExtraArg (8).c_str();
  std::string pathNumbers = manager.getExtraArg (9);
  // number of luminance values
  uint numL = manager.numExtraArgs() - 13;
  // total number of distractors
  Image<PixRGB<byte> >* distractor[3][numL], *target[numL];
  // read the distractor patches
  for (uint j = 0; j < 3; j++)
    {
      const char* distrName = manager.getExtraArg (10 + j).c_str();
      for (uint i = 0; i < numL; i++)
        {
          char itemName[30];
          sprintf(itemName, "%s_%s.ppm", distrName,
                  manager.getExtraArg (13 + i).c_str());
          distractor[j][i] = new Image<PixRGB<byte> >
            (Raster::ReadRGB(itemName));
        }
    }
  // read target patches
  for (uint i = 0; i < numL; i++)
    {
      char itemName[30];
      sprintf(itemName, "%s_%s.ppm", targetName,
              manager.getExtraArg (13 + i).c_str());
      target[i] = new Image<PixRGB<byte> >
        (Raster::ReadRGB(itemName));
    }
  int pw = target[0]->getWidth(), ph = target[0]->getHeight();

  // ------------------ INITIALIZE PARAMS -------------------------
  for (uint j = 0; j < 3; j++)
    for (uint i = 0; i < numL; i++)
      {
        //all distractors equal size
        ASSERT(pw == distractor[j][i]->getWidth() &&
               ph == distractor[j][i]->getHeight());
      }

  initRandomNumbers();

  // initialize results:
  Image<PixRGB<byte> > image(w, w, ZEROS);
  Image<PixRGB<byte> > randomGrid(w, w, ZEROS);
  Image<byte> targets(w, w, ZEROS);

  int cols = int(w / (FACTOR * pw)),
    rows = int(w / (FACTOR * ph)),
    numBox = cols * rows;
  int numPerD = int((numBox - 1) / 3);
  int remainder = (numBox - 1) % 3;
  LINFO (" image grid: %d x %d, number per distractor type = %d, "
         " remainder = %d",
         cols, rows, numPerD, remainder);

  // read the 2 digit randomNumber patches: to avoid any head
  // movements while entering the digits, we restrict each digit to
  // 1..5, hence a max of 25 numbers
  int numR, numRows, numCols;
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
    }
  Image<PixRGB<byte> >* numbers[numR];
  std::vector <std::string> names;

  for (int i = 1; i <= numRows; i ++)
    for (int j = 1; j <= numCols; j++)
      {
        int k = (i - 1) * numCols + j - 1;
        names.push_back (pathNumbers + "/" + toStr (i) + toStr (j) +
                         ".ppm");
        numbers[k] = new Image<PixRGB<byte> >
          (Raster::ReadRGB(names[k]));
        //all numbers equal size
        ASSERT(pw == numbers[k]->getWidth() &&
               ph == numbers[k]->getHeight());
      }


  // --------------- MAKE THE SEARCH ARRAY AND RANDOM GRID --------------
  // keep a list of all locns in the grid where items can appear
  int locn[numBox][2];
  // we will first create a dummy with values 1...numBox and then
  // randomize it and allocate items according to its randomized
  // contents
  int dummy[numBox];

  // list of indices that will be randomized for flashing numbers
  int random[numBox];
  FILE* f = fopen (reco, "w");
  FILE* check = fopen (checkResponse, "w");
  uint idx = 0;
  for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
      {
        int k = i * cols + j;
        dummy[k] = k; // intialize with index
        // the above item is at locn. <i+1, j+1> in the grid
        locn[k][0] = i + 1;
        locn[k][1] = j + 1;

        // random number to be flashed is btw 0..24
        if (idx == 25)
          idx = 0;
        random[k] = idx; // intialize now: randomize later
        idx ++;
      }

  // shuffle the grid to randomize the locn of items
  randShuffle (dummy, numBox);

  // also randomize the display of numbers in the grid
  randShuffle (random, numBox);

  // allocate the target to the randomized contents of dummy
  idx = 0; // index for the box and random array
  int r = locn[dummy[idx]][0];
  int c = locn[dummy[idx]][1];
  int center_x = 0;
  int center_y = 0;
  // find a random luminance for the target
  int randomL = dummy[idx] % numL;
  LINFO("-- TARGET, lum %d AT (%d, %d)", randomL, r, c);
  image_patch (*target[randomL], c-1, r-1, image, alpha, FACTOR, targets, 1,
               randomGrid, *numbers[random[idx]], center_x, center_y);
  fprintf (f, "%d %d %d %d %s_%d.ppm\n", r, c, center_x, center_y,
           targetName, randomL);

  // record the random number at the target locn to compare response
  int start = names[random[idx]].rfind ("/");
  int end = names[random[idx]].rfind (".ppm");
  fprintf (check, "%s", names[random[idx]].
           substr (start+1, end-start-1).c_str());
  fclose (check);
  idx++;

  // allocate the items to the randomized contents of dummy
  for (uint disType = 0; disType < 3; disType ++)
    for (int count = 0; count < numPerD; count ++)
      {
        r = locn[dummy[idx]][0];
        c = locn[dummy[idx]][1];
        // find a random luminance for the target
        int randomL = dummy[idx] % numL;
        LINFO("-- distractor %d, lum %d AT (%d, %d)", disType + 1, randomL,
              r, c);
        image_patch (*distractor[disType][randomL], c-1, r-1,
                     image, alpha, FACTOR,
                     targets, 0, randomGrid, *numbers[random[idx]],
                     center_x, center_y);
        fprintf (f, "%d %d %d %d %s_%d.ppm\n", r, c, center_x, center_y,
                 manager.getExtraArg(10 + disType).c_str(), randomL);
        idx++;
      }
  // allocate the remaining items (if any) to the randomized boxes
  uint disType = 0;
  for (int count = 0; count < remainder; count ++, disType ++)
    {
        r = locn[dummy[idx]][0];
        c = locn[dummy[idx]][1];
        // find a random luminance for the target
        int randomL = dummy[idx] % numL;
        LINFO("-- distractor %d, lum %d AT (%d, %d)", disType + 1, randomL,
              r, c);
        image_patch (*distractor[disType][randomL], c-1, r-1,
                     image, alpha, FACTOR,
                     targets, 0, randomGrid, *numbers[random[idx]],
                     center_x, center_y);
        fprintf (f, "%d %d %d %d %s_%d.ppm\n", r, c, center_x, center_y,
                 manager.getExtraArg(10 + disType).c_str(), randomL);
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
  Raster::WriteGray(targets, output, RASFMT_PNM);

  // erase the memory used
  for (uint j = 0; j < 3; j++)
    for (uint i = 0; i < numL; i++)
      delete (distractor[j][i]);

  for (int i = 0; i < numR; i++)
    delete (numbers[i]);
  manager.stop();
  return 0;
}

// ######################################################################
void image_patch(const Image< PixRGB<byte> >& patch, const int ti,
                 const int tj, Image< PixRGB<byte> >& image,
                 const double alpha,  const float FACTOR,
                 Image<byte>& targets,
                 bool do_target, Image< PixRGB<byte> >& randomGrid,
                 const Image< PixRGB<byte> >& randomNum,
                 int& center_x, int& center_y)
{
  int pw = patch.getWidth(), ph = patch.getHeight();
  int w = image.getWidth();

  int jitx = int(randomDouble() * (FACTOR - 2.0) * pw);
  int jity = int(randomDouble() * (FACTOR - 2.0) * ph);

  float jita = float(alpha * 3.14159 / 180.0 * (randomDouble() - 0.5) * 2.0);
  int offset = int(w - floor(w / (pw * FACTOR)) * (pw * FACTOR)) / 2;

  PixRGB<byte> zero(0, 0, 0);
  int px = 0, py = 0;
  int minx = w, maxx = 0, miny = w, maxy = 0;

  for (double y = int(tj * ph * FACTOR); y < int(tj * ph * FACTOR + ph); y ++)
    {
      for (double x = int(ti * pw * FACTOR); x < int(ti * pw * FACTOR + pw);
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
              // find the extremities of this item
              minx = std::min (minx, x2);
              maxx = std::max (maxx, x2);
              miny = std::min (miny, y2);
              maxy = std::max (maxy, y2);
            }
          px ++;
        }
      py ++;
      px = 0;
    }
  // find the center of this item
  center_x = (minx + maxx) / 2;
  center_y = (miny + maxy) / 2;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
