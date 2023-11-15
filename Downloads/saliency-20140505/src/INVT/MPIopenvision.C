/*!@file INVT/MPIopenvision.C  MPI version of openvision.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/MPIopenvision.C $
// $Id: MPIopenvision.C 10845 2009-02-13 08:49:12Z itti $
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

#include <cmath>
#include <fstream>
#ifdef HAVE_MPI_H
#include <mpi.h>
#endif

#define ITMAX 200
#define FTOL 0.00000001f
#define NBCOEFFS 144
#define MASTER 0
#define MSG_DATA 1000
#define MSG_RESULT 2000
#define PATH "/tmphpc-01/itti/openvision"
#define NB_PIC 14
#define NB_FILTERS 3

#define NB_EVALS 1

#define NB_PROC (NBCOEFFS * NB_EVALS + 1)


int main(int argc, char **argv)
{
#ifndef HAVE_MPI_H

  LFATAL("<mpi.h> must be installed to use this program");

#else

  // Set verbosity:
  int loglev = LOG_ERR;
  MYLOGVERB = loglev;

  // Number of coefficients:
  int n = 4;
  int m = n * n * NB_FILTERS * 3;

  // Initialize the MPI system and get the number of processes:
  int myrank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Check for proper number of processes:
  if(size < NB_PROC)
    {
      LERROR("*** Error: %d Processes needed. ***", NB_PROC);
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

      // Generate the starting point, directions and function value:
      float p[m];
      float xi[m][m];
      for(int i = 0; i < m; i++)
        {
          p[i] = 0.0f;
          for(int j = 0; j < m; j++)
            xi[j][i] = 0.0f;
          xi[i][i] = 1.0f;
        }
      p[0] = 0.1f;
      float fp = 10.0f;

      // **************** OPTIMIZATION ****************

      for (int iter = 0; ; iter++)
        {
          LERROR("### ITERATION %i ###", iter);
          int stop = 0;
          float del1 = 0.0f;
          float fp1 = fp;
          int ibig1 = 0;
          float p1[m];
          for (int part = 0; part < 2; part++)
            {
              // Sent data to slave processes:
              for (int i = 0; i < m; i++)
                for (int mu = 1; mu <= NB_EVALS; mu++)
                  {
                    int proc = (i * NB_EVALS) + mu;
                    float pt[m];
                    for (int j = 0; j < m; j++)
                      if (part == 0)
                        pt[j] = p[j] + (mu * 0.1f * xi[j][i]);
                      else
                        pt[j] = p[j] - (mu * 0.1f * xi[j][i]);
                    MPI_Send(&stop, 1, MPI_INT, proc,
                             4 * iter + part,
                             MPI_COMM_WORLD);
                    MPI_Send(pt, m, MPI_FLOAT, proc,
                             MSG_DATA + 4 * iter + part,
                             MPI_COMM_WORLD);
                  }
              LERROR("*** PART %i : data sent to slaves.",
                     part + 1);
              // Receive results and keep the best:
              for (int i = 0; i < m; i++)
                for (int mu = 1; mu <= NB_EVALS; mu++)
                  {
                    int proc = (i * NB_EVALS) + mu;
                    float fpt;
                    MPI_Recv(&fpt, 1, MPI_FLOAT, proc,
                             MSG_RESULT + 4 * iter + part,
                             MPI_COMM_WORLD, &status);
                    if (fp - fpt > del1)
                      {
                        del1 = fp - fpt;
                        ibig1 = i;
                        for (int j = 0; j < m; j++)
                          if (part == 0)
                            p1[j] = p[j] + (mu * 0.1f * xi[j][i]);
                          else
                            p1[j] = p[j] - (mu * 0.1f * xi[j][i]);
                        fp1 = fpt;
                      }
                  }
              LERROR("*** PART %i : result received from slaves.",
                     part + 1);
            }
          float del2 = 0.0f;
          float fp2 = fp1;
          float p2[m];
          int ibig2 = 0;
          for (int part = 0; part < 2; part++)
            {
              // Sent data to slave processes:
              for (int i = 0; i < m; i++)
                for (int mu = 1; mu <= NB_EVALS; mu++)
                  {
                    int proc = (i * NB_EVALS) + mu;
                    float pt[m];
                    for (int j = 0; j < m; j++)
                          if (part == 0)
                            pt[j] = p1[j] + (mu * 0.1f * xi[j][i]);
                          else
                            pt[j] = p1[j] - (mu * 0.1f * xi[j][i]);
                    MPI_Send(&stop, 1, MPI_INT, proc,
                             4 * iter + 2 + part,
                             MPI_COMM_WORLD);
                    MPI_Send(pt, m, MPI_FLOAT, proc,
                             MSG_DATA + 4 * iter + 2 + part,
                             MPI_COMM_WORLD);
                  }
              LERROR("*** PART %i : data sent to slaves",
                     part + 3);
              // Receive results and keep the best:
              for (int i = 0; i < m; i++)
                for (int mu = 1; mu <= NB_EVALS; mu++)
                  {
                    int proc = (i * NB_EVALS) + mu;
                    float fpt;
                    MPI_Recv(&fpt, 1, MPI_FLOAT, proc,
                             MSG_RESULT + 4 * iter + 2 + part,
                             MPI_COMM_WORLD, &status);
                    if (fp1 - fpt > del2)
                      {
                        del2 = fp1 - fpt;
                        ibig2 = i;
                        for (int j = 0; j < m; j++)
                          if (part == 0)
                            p2[j] = p1[j] + (mu * 0.1f * xi[j][i]);
                          else
                            p2[j] = p1[j] - (mu * 0.1f * xi[j][i]);
                        fp2 = fpt;
                      }
                  }
              LERROR("*** PART %i : result received from slaves.",
                     part + 3);
            }

          LERROR("old value = %f *** new value = %f.", fp, fp2);

          // Exit the loop if ITMAX is reach or if the decrease is too low:
          if (2.0 * fabs(fp - fp2) <= FTOL * (fabs(fp) + fabs(fp2)))
            {
              stop = 1;
              for (int i = 1; i < NB_PROC; i++)
                {
                  MPI_Send(&stop, 1, MPI_INT, i,
                           4 * iter + 4, MPI_COMM_WORLD);
                }
              LERROR("### Low decrease ! ###");
              MPI_Finalize();
              return 0;
            }
          if (iter == ITMAX)
            {
              stop = 1;
              for (int i = 1; i < NB_PROC; i++)
                {
                  MPI_Send(&stop, 1, MPI_INT, i,
                           4 * iter + 4, MPI_COMM_WORLD);
                }
              LERROR("### Maximum iterations exceeded ! ###");
              MPI_Finalize();
              return 0;
            }

          // Update data:
          float xit[m];
          float norm = 0;
          int ibig;
          if (del1 > del2)
            ibig = ibig1;
          else
            ibig = ibig2;
          for (int j = 0; j < m; j++)
            {
              xit[j] = p2[j] - p[j];
              norm += xit[j] * xit[j];
              p[j] = p2[j]; // Current position updated
            }

          const std::string filename =
            sformat("%s/results/step%03i.txt", PATH, iter);
          std::ofstream resultfile (filename.c_str());
          if (resultfile.is_open())
            {
              for (int j = 0; j < m; j++)
                resultfile << p[j] << "\n";
              resultfile.close();
            }

          norm = sqrt(norm);
          for (int j = 0; j < m; j++)
            {
              xi[j][ibig] = xit[j] / norm; // Current directions updated
            }
          fp = fp2; // Current value updated
        }
    }
  else
    {
      // ################ SLAVES PROCESS ################

      // **************** INITIALIZATION ****************

      MPI_Status status;

      // Generate the haar transform matrix:
      Image<float> hmat(n, n, NO_INIT);
      for(int i = 0; i < n; i++)
        {
          hmat.setVal(i, 0, 1.0f);
        }
      for(int i = 0; i < n / 2; i++)
        {
          hmat.setVal(i, 1, 1.0f);
          hmat.setVal(i + n / 2, 1, -1.0f);
          hmat.setVal(2 * i, i + n / 2, 1.0f);
          hmat.setVal(2 * i + 1, i + n / 2, -1.0f);
        }

      // Load the input images and masks:
      ImageSet< PixRGB<byte> > input(NB_PIC);
      ImageSet<byte> mask(NB_PIC);
      for (int i = 0; i < NB_PIC; i++)
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
                   iter, MPI_COMM_WORLD, &status);
          if (stop == 1)
            {
              MPI_Finalize();
              return 0;
            }

          // Receive data from master:
          float data[m];
          MPI_Recv(data, m, MPI_FLOAT, MASTER,
                   MSG_DATA + iter, MPI_COMM_WORLD, &status);

          // Convert data into filters:
          ImageSet<float> trans(NB_FILTERS * 3);
          for (int i = 0; i < NB_FILTERS * 3; i++)
            trans[i] = Image<float>(data + (n * n * i), n, n);
          ImageSet<float> filter(NB_FILTERS * 3);
          Dims filterdim(8, 8);
          for (int i = 0; i < NB_FILTERS * 3; i++)
            filter[i] = scaleBlock(matrixMult(hmat,
                                              matrixMult(trans[i],
                                                         transpose(hmat))),
                                   filterdim);

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

          // Init resut:
          float result = 0.0f;

          // Compute the error for each image:
          for (int p = 0; p < NB_PIC; p++)
            {
              // Instantiate a ModelManager:
              ModelManager manager("Open Attention Model");

              // Instantiate our various ModelComponents:
              nub::soft_ref<RawVisualCortex> vcx(new RawVisualCortex(manager));
              manager.addSubComponent(vcx);

              for (int i = 0; i < 1; i++)
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
              vcx->input(InputFrame::fromRgb(&input[p]));

              // Get the resulting saliency map:
              Image<float> sm = vcx->getOutput();

              // Normalize the saliency map:
              inplaceNormalize(sm, 0.0f, 255.0f);
              Image<byte> smb = sm;

              // Blur and resize the binary mask:
              Dims smbdim = smb.getDims();
              Image<byte> blur_mask = binaryReverse(chamfer34(mask[p],
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
              float detect_coeff = 1.0f - ((max_in - max_out) / 255.0f);
              result += detect_coeff * detect_coeff;

              // Stop all our ModelComponents
              manager.stop();
            }

          // Send the result to the master :
          result /= NB_PIC;
          result += fabs(dotprod) * 0.001f;
          MPI_Send(&result, 1, MPI_FLOAT, MASTER,
                   MSG_RESULT + iter, MPI_COMM_WORLD);
        }
    }

#endif // HAVE_MPI_H

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
