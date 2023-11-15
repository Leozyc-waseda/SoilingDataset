/*!@file INVT/MPIopenvision2.C  another MPI version of openvision.C */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/MPIopenvision2.C $
// $Id: MPIopenvision2.C 10845 2009-02-13 08:49:12Z itti $
//

#include "Channels/RGBConvolveChannel.H"
#include "Component/ModelManager.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Pixels.H"
#include "Image/Transforms.H"
#include "Media/FrameSeries.H"
#include "Neuro/NeuroOpts.H"
#include "Channels/RawVisualCortex.H"
#include "Raster/Raster.H"
#include "Util/sformat.H"

#include <fstream>
#ifdef HAVE_MPI_H
#include <mpi.h>
#endif

#define ITMAX 200
#define FTOL 0.000001f
#define MASTER 0
#define MSG_STOP 10
#define MSG_PIC 1
#define MSG_DATA 2
#define MSG_RESULT 200
#define PATH "/tmphpc-01/itti/openvision"
#define INPUTFILE "/tmphpc-01/itti/openvision/start.txt"
#define NB_PICS 14
#define NB_FILTERS 3
#define NCOEFFS 8
#define MCOEFFS NCOEFFS * NCOEFFS * NB_FILTERS * 3
#define NB_EVALS 10
#define NB_WAVES 5
#define NB_PROCS NB_PICS * NB_EVALS * 2 + 1

int main(int argc, char **argv)
{
#ifndef HAVE_MPI_H

  LFATAL("<mpi.h> must be installed to use this program");

#else

  // Set verbosity:
  int loglev = LOG_ERR;
  MYLOGVERB = loglev;

  // Number of coefficients:
  int n = NCOEFFS;
  int m = MCOEFFS;

  // Initialize the MPI system and get the number of processes:
  int myrank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Check for proper number of processes:
  if(size < NB_PROCS)
    {
      LERROR("*** Error: %i Processes needed. ***", NB_PROCS);
      MPI_Finalize();
      return 1;
    }

  // Get the rank of the process:
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (myrank == MASTER)
    {
      // ################ MASTER PROCESS ################

      // **************** INITIALIZATION ****************

      MPI_Status status;

      // Load start coefficients:
      float p[m];
      std::ifstream inputfile (INPUTFILE);
      if (inputfile.is_open())
        {
          for (int j = 0; j < m; j++)
            inputfile >> p[j];
          inputfile.close();
        }
      else
        {
          LERROR("*** Cannot open input file !");
          return 1;
        }

      // Generate the starting point, directions and function value:
      float xi[m][m];
      for(int i = 0; i < m; i++)
        {
          for(int j = 0; j < m; j++)
            xi[j][i] = 0.0f;
          xi[i][i] = 1.0f;
        }
      float fp = 10.0f;

      // **************** OPTIMIZATION ****************

      for (int iter = 0; ; iter++)
        {
          LERROR("### ITERATION %i ###", iter);

          int stop = 0;
          float del = 0.0f;
          int ibig = 0;
          float fstart = fp;
          float start[m];
          for (int i = 0; i < m; i++)
            start[m] = p[m];
          float fpt = fp;
          int mubig = 0;

          // Loop over all directions:
          for (int i = 0; i < m; i++)
            {
              mubig = 0;
              for (int wave = 0; wave < NB_WAVES; wave++)
                {
                  int muplus = NB_EVALS * wave;
                  // Send data to slaves:
                  for (int part = 0; part < 2; part++)
                    for (int mu = 1; mu <= NB_EVALS; mu++)
                      {
                        int mu2 = (1 - 2 * part) * (mu + muplus);
                        float ptry[m];
                        for (int j = 0; j < m; j++)
                          ptry[j] = p[j] + (mu2 * 0.02f * xi[j][i]);
                        for (int pic = 0; pic < NB_PICS; pic++)
                          {
                            int proc = (NB_EVALS * NB_PICS * part);
                            proc += (NB_PICS * (mu - 1)) + pic + 1;
                            MPI_Send(&stop, 1, MPI_INT, proc,
                                     MSG_STOP, MPI_COMM_WORLD);
                            MPI_Send(&pic, 1, MPI_INT, proc,
                                     MSG_PIC, MPI_COMM_WORLD);
                            MPI_Send(ptry, m, MPI_FLOAT, proc,
                                     MSG_DATA, MPI_COMM_WORLD);

                          }
                      }
                  LERROR("*** Data sent to slaves");

                  // Receive result from slaves and keep best decrease:
                  for (int part = 0; part < 2; part++)
                    for (int mu = 1; mu <= NB_EVALS; mu++)
                      {
                        float fptry = 0.0f;
                        for (int pic = 0; pic < NB_PICS; pic++)
                          {
                            float value;
                            int proc = (NB_EVALS * NB_PICS * part);
                            proc += (NB_PICS * (mu - 1)) + pic + 1;
                            MPI_Recv(&value, 1, MPI_FLOAT, proc,
                                     MSG_RESULT, MPI_COMM_WORLD,
                                     &status);
                            fptry += value;
                          }
                        fptry /= NB_PICS;
                        if (fptry < fpt)
                          {
                            fpt = fptry;
                            mubig = (1 - 2 * part) * (mu + muplus);
                          }
                      }
                }

              LERROR("*** Result received");
              // Keep the greatest decrease:
              if (fp - fpt > del)
                {
                  del = fp - fpt;
                  ibig = i;
                }
              fp = fpt;
              for (int j = 0; j < m; j++)
                p[j] += mubig * 0.02f * xi[j][i];

              LERROR("*** Direction %i : mu = %i", i, mubig);
              LERROR("*** New value : %f", fp);

//               float maxcoeff1 = 0;
//               float maxcoeff2 = 0;
//               float maxcoeff3 = 0;
//               for (int j = 0; j < m / 3; j++)
//                 {
//                   if (fabs(p[j]) > maxcoeff1)
//                     maxcoeff1 = fabs(p[j]);
//                   if (fabs(p[j + m / 3]) > maxcoeff2)
//                     maxcoeff2 = fabs(p[j + m / 3]);
//                   if (fabs(p[j + 2 * m / 3]) > maxcoeff3)
//                     maxcoeff3 = fabs(p[j + 2 * m / 3]);
//                 }
//               for (int j = 0; j < m / 3; j++)
//                 {
//                   p[j] /= maxcoeff1;
//                   p[j + m / 3] /= maxcoeff2;
//                   p[j + 2 * m / 3] /= maxcoeff3;
//                 }
             }

          LERROR("### Total decrease for iteration : %f", fstart - fp);

          // Save current coefficients:
          const std::string filename =
            sformat("%s/results/step%03i.txt", PATH, iter);
          std::ofstream resultfile (filename.c_str());
          if (resultfile.is_open())
            {
              for (int j = 0; j < m; j++)
                resultfile << p[j] << "\n";
              resultfile.close();
            }
          LERROR("### Current result saved in %s", filename.c_str());

          // Exit the loop if ITMAX is reach or if the decrease is too low:
          if (2.0 * fabs(fstart - fp) <= FTOL * (fabs(fstart) + fabs(fp)))
            {
              stop = 1;
              for (int i = 1; i < NB_PROCS; i++)
                MPI_Send(&stop, 1, MPI_INT, i,
                         MSG_STOP, MPI_COMM_WORLD);
              LERROR("### Low decrease ! ###");
              sleep(10);
              // MPI_Finalize();
              return 0;
            }
          if (iter == ITMAX)
            {
              stop = 1;
              for (int i = 1; i < NB_PROCS; i++)
                MPI_Send(&stop, 1, MPI_INT, i,
                         MSG_STOP, MPI_COMM_WORLD);
              LERROR("### Maximum iterations exceeded ! ###");
              sleep(10);
              // MPI_Finalize();
              return 1;
            }

          // Update data:
          float xit[m];
          float norm = 0;
          for (int j = 0; j < m; j++)
            {
              xit[j] = p[j] - start[j];
              norm += xit[j] * xit[j];
            }
          norm = sqrt(norm);
          for (int j = 0; j < m; j++)
            {
              xi[j][ibig] = xit[j] / norm;
            }
        }
    }
  else
    {
      // ################ SLAVES PROCESS ################

      // **************** INITIALIZATION ****************

      MPI_Status status;

      // Generate the haar transform matrix:
      Image<float> hmat(n, n, ZEROS);
      for(int i = 0; i < n; i++)
        {
          hmat.setVal(i, 0, 1.0f);
        }
      for(int i = 0; i < n / 2; i++)
        {
          hmat.setVal(i, 1, 1.0f);
          hmat.setVal(i + n / 2, 1, -1.0f);
          if (i - 2 < 0)
            {
              hmat.setVal(i, 2, 1.0f);
              hmat.setVal(i + 2, 2, -1.0f);
            }
          else
            {
              hmat.setVal(i + 2, 3, 1.0f);
              hmat.setVal(i + 4, 3, -1.0f);
            }
          hmat.setVal(2 * i, i + n / 2, 1.0f);
          hmat.setVal(2 * i + 1, i + n / 2, -1.0f);
        }

      // Load the input images and masks:
      ImageSet< PixRGB<byte> > input(NB_PICS);
      ImageSet<byte> mask(NB_PICS);
      for (int i = 0; i < NB_PICS; i++)
        {
          input[i] =
            Raster::ReadRGB(sformat("%s/pictures/PIC_%03i.PPM", PATH, i));
          mask[i] =
            Raster::ReadGray(sformat("%s/pictures/PIC_%03i.PGM", PATH, i));
        }

      LERROR("### SLAVE %i INITIALIZED SUCCESFULLY ! ###", myrank);

      // **************** OPTIMIZATION ****************

      for (int iter = 0; ; ++iter)
        {
          // Receive the stop message:
          int stop;
          MPI_Recv(&stop, 1, MPI_INT, MASTER,
                   MSG_STOP, MPI_COMM_WORLD, &status);
          if (stop == 1)
            {
              // MPI_Finalize();
              return 0;
            }

          // Receive data from master:
          int pic;
          MPI_Recv(&pic, 1, MPI_INT, MASTER,
                   MSG_PIC, MPI_COMM_WORLD, &status);
          float data[m];
          MPI_Recv(data, m, MPI_FLOAT, MASTER,
                   MSG_DATA, MPI_COMM_WORLD, &status);

          // Convert data into filters:
          ImageSet<float> trans(NB_FILTERS * 3);
          for (int i = 0; i < NB_FILTERS * 3; i++)
            trans[i] = Image<float>(data + (n * n * i), n, n);
          ImageSet<float> filter(NB_FILTERS * 3);
          for (int i = 0; i < NB_FILTERS * 3; i++)
            filter[i] = matrixMult(transpose(hmat),
                                   matrixMult(trans[i], hmat));

          // Compute the dot product of the filters:
          Image<float> prod1(8, 8, ZEROS);
          Image<float> prod2(8, 8, ZEROS);
          Image<float> prod3(8, 8, ZEROS);
          for (int i = 0; i < NB_FILTERS; i++)
            {
              prod1 += filter[3 * i] * filter[3 * i + 1];
              prod2 += filter[3 * i] * filter[3 * i + 2];
              prod3 += filter[3 * i + 2] * filter[3 * i + 1];
            }
          float dotprod = sum(prod1) + sum(prod2) + sum(prod3);

          // Instantiate a ModelManager:
          ModelManager manager("Open Attention Model");

          // Instantiate our various ModelComponents:
          nub::soft_ref<RawVisualCortex> vcx(new RawVisualCortex(manager));
          manager.addSubComponent(vcx);

          for (int i = 0; i < NB_FILTERS; i++)
            {
              // Create a channel attached to each filter:
              nub::soft_ref<RGBConvolveChannel>
                channel(new RGBConvolveChannel(manager));
              channel->setDescriptiveName(sformat("RGBConvolve%d", i));
              channel->setTagName(sformat("rgbconv%d", i));

              // Assign the 3 filters to the channel:
              channel->setFilters(filter[3 * i], filter[3 * i + 1],
                                  filter[3 * i + 2],
                                  CONV_BOUNDARY_ZERO);

              // Attach the channel to our visual cortex:
              vcx->addSubChan(channel);
            }

          // Let's get all our ModelComponent instances started:
          manager.exportOptions(MC_RECURSE);
          manager.setModelParamString("UsingFPE", "false");
          manager.setModelParamString("TestMode", "true");

          // reduce amount of log messages:
          MYLOGVERB = loglev;

          manager.start();

          // Process the image through the visual cortex:
          vcx->input(InputFrame::fromRgb(&input[pic]));

          // Get the resulting saliency map:
          Image<float> sm = vcx->getOutput();

          // Normalize the saliency map:
          inplaceNormalize(sm, 0.0f, 255.0f);
          Image<byte> smb = sm;

          // Blur and resize the binary mask:
          Dims smbdim = smb.getDims();
          Image<byte> blur_mask = binaryReverse(chamfer34(mask[pic],
                                                          (byte) 255),
                                                (byte) 255);
          inplaceLowThresh(blur_mask, (byte) 200);
          Image<byte> mask_in = scaleBlock(blur_mask, smbdim);
          Image<byte> mask_out = binaryReverse(mask_in, (byte) 255);

          // Weight the saliency map using the in and out masks:
          Image<float> smb_in = (mask_in * (1.0f / 255.0f)) * smb;
          Image<float> smb_out = (mask_out * (1.0f / 255.0f)) * smb;

          // Get the max_in and max_out values:
          float max_in, max_out, min;
          getMinMax(smb_in, min, max_in);
          getMinMax(smb_out, min, max_out);

          // Compute the error:
          float error = 1.0f - ((max_in - max_out) / 255.0f);
          float result = error * error + 0.001f * fabs(dotprod);

          // Stop all our ModelComponents
          manager.stop();

          // Send the result to the master :
          MPI_Send(&result, 1, MPI_FLOAT, MASTER,
                   MSG_RESULT, MPI_COMM_WORLD);
        }
    }

#endif // HAVE_MPI_H

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

