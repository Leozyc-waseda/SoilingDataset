/*!@file GA/GAMPIvision.C Learning filters with genetic algorithm */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/GA/GAMPIvision.C $
// $Id: GAMPIvision.C 10845 2009-02-13 08:49:12Z itti $
//

#include "Channels/ChannelOpts.H"
#include "Channels/ConvolveChannel.H"
#include "Component/ModelManager.H"
#include "GA/GAPopulation.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Pixels.H"
#include "Channels/RawVisualCortex.H"
#include "Raster/Raster.H"

#include <fstream>
#ifdef HAVE_MPI_H
#include <mpi.h>
#endif

#define ITMAX 400
#define MASTER 0
#define MSG_DATA 0
#define MSG_RESULT 1
#define MSG_STOP 2
#define PATH "/home/hpc-26/itti/openvision"
#define NB_PICS 10
#define NB_FILTERS 3
#define NCOEFFS 3
#define POP_SIZE 196
#define RAD 64

int main(const int argc, const char **argv)
{
#ifndef HAVE_MPI_H

  LFATAL("<mpi.h> must be installed to use this program");

#else

  // Set verbosity :
  int loglev = LOG_ERR;
  MYLOGVERB = loglev;

  // Number of coefficients :
  int n = 1 << NCOEFFS;
  int csize = n * n * NB_FILTERS;
  int psize = POP_SIZE;

  // Initialize the MPI system and get the number of processes :
  int myrank, nodes;
  MPI_Init(const_cast<int *>(&argc), const_cast<char ***>(&argv));
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);

  // Check for proper number of processes :
  if(nodes < 2)
    {
      LERROR("*** At least two nodes needed ***");
      MPI_Finalize();
      return 1;
    }

  // Get the rank of the process :
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (myrank == MASTER)
    {
      // ################ MASTER PROCESS ################

      MPI_Status status;
      int stop = 0;

      // Create a new population :
      LERROR("### START ! ###");
      GAPopulation pop(psize, csize);

      for (int iter = 1; ; iter++)
        {
          // Compute the fitness of every chromosome :
          int current = 0;
          int done = 0;
          float low_fit = 1000.0f;
          float high_fit = 0.0f;
          while (psize - done > 0)
            {
              int max;
              if (psize - done > nodes - 1)
                max = nodes;
              else
                max = psize - done + 1;
              for (int i = MASTER + 1; i < max; i++)
                {
                  GAChromosome c = pop.get_chromosome(current);
                  int genes[csize];
                  for (int j = 0; j < csize; j++)
                    genes[j] = c.get_gene(j);
                  MPI_Send(&stop, 1, MPI_INT, i, MSG_STOP,
                           MPI_COMM_WORLD);
                  MPI_Send(genes, csize, MPI_INT, i, MSG_DATA,
                           MPI_COMM_WORLD);
                  current++;
                }
              for (int i = MASTER + 1; i < max; i++)
                {
                  float fit;
                  GAChromosome c = pop.get_chromosome(done);
                  MPI_Recv(&fit, 1, MPI_FLOAT, i, MSG_RESULT,
                           MPI_COMM_WORLD, &status);
                  c.set_fitness(fit);
                  pop.set_chromosome(done, c);
                  done++;
                  if (fit > high_fit)
                    high_fit = fit;
                  if (fit < low_fit)
                    low_fit = fit;
                }
            }

          // Compute mean fitness and standard deviation :
          pop.compute_pop_fitness();
          pop.compute_sigma();

          // Display various stuff :
          LERROR("*** Highest fitness : %f", high_fit);
          LERROR("*** Lowest fitness : %f", low_fit);
          LERROR("*** Mean fitness : %f", pop.get_mean_fitness());
          LERROR("*** Standard deviation : %f", pop.get_sigma());

          // Select population :
          pop.linear_scaling();
          pop.selection();

          // Save current population :
          char filename[1024];
          sprintf(filename, "%s/results/pop%03i.txt", PATH, iter);
          std::ofstream resultfile (filename);
          if (resultfile.is_open())
            {
              resultfile << pop;
              resultfile.close();
            }

          // Population evolution :
          LERROR("### EVOLUTION %i ###", iter);
          pop.crossover();
          pop.mutate();
          pop.update();

          // Exit the loop if ITMAX is reach :
          if (iter == ITMAX)
            {
              stop = 1;
              for (int i = 1; i < nodes; i++)
                MPI_Send(&stop, 1, MPI_INT, i,
                         MSG_STOP, MPI_COMM_WORLD);
              LERROR("### Maximum evolutions exceeded ! ###");
              sleep(10);
              // MPI_Finalize();
              return 1;
            }

          // TODO : add a test to exit when the evolution is stalled.

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

      // Load the input images and the targets locations :

      // Set of pictures :
      ImageSet< PixRGB<byte> > input(NB_PICS);

      // Targets locations for each subjects and pictures :
      Point2D<int> ch[20][NB_PICS];
      Point2D<int> kp[20][NB_PICS];
      Point2D<int> pw[20][NB_PICS];
      Point2D<int> wkm[20][NB_PICS];

      // Numbers of targets for each subject and pictures :
      int nch[NB_PICS];
      int nkp[NB_PICS];
      int npw[NB_PICS];
      int nwkm[NB_PICS];

      // Data loading :
      for (int i = 0; i < NB_PICS; i++)
        {
          char framename[1024];
          char chname[1024];
          char kpname[1024];
          char pwname[1024];
          char wkmname[1024];
          sprintf(framename,
                  "%s/pictures/sat_%03i.ppm", PATH, i);
          sprintf(chname,
                  "%s/pictures/sat_%03i.ch.dat", PATH, i);
          sprintf(kpname,
                  "%s/pictures/sat_%03i.kp.dat", PATH, i);
          sprintf(pwname,
                  "%s/pictures/sat_%03i.pw.dat", PATH, i);
          sprintf(wkmname,
                  "%s/pictures/sat_%03i.wkm.dat", PATH, i);
          input[i] = Raster::ReadRGB(framename);
          int count = 0;
          std::ifstream chfile(chname);
          bool eofile = false;
          while (!chfile.eof())
            {
              int px, py;
              chfile >> px >> py;
              if (!chfile.eof())
                {
                  ch[count][i].i = px;
                  ch[count][i].j = py;
                  count++;
                }
              else
                eofile = true;
            }
          chfile.close();
          nch[i] = count;
          count = 0;
          std::ifstream kpfile(kpname);
          eofile = false;
          while (!eofile)
            {
              int px, py;
              kpfile >> px >> py;
              if (!kpfile.eof())
                {
                  kp[count][i].i = px;
                  kp[count][i].j = py;
                  count++;
                }
              else
                eofile = true;
            }
          kpfile.close();
          nkp[i] = count;
          count = 0;
          std::ifstream pwfile(chname);
          eofile = false;
          while (!eofile)
            {
              int px, py;
              pwfile >> px >> py;
              if (!pwfile.eof())
                {
                  pw[count][i].i = px;
                  pw[count][i].j = py;
                  count++;
                }
              else
                eofile = true;
            }
          pwfile.close();
          npw[i] = count;
          count = 0;
          std::ifstream wkmfile(chname);
          eofile = false;
          while (!eofile)
            {
              int px, py;
              wkmfile >> px >> py;
              if (!wkmfile.eof())
                {
                  wkm[count][i].i = px;
                  wkm[count][i].j = py;
                  count++;
                }
              else
                eofile = true;
            }
          wkmfile.close();
          nwkm[i] = count;
        }

      // **************** OPTIMIZATION ****************

      for (int iter = 0; ; ++iter)
        {
          // Receive the stop message :
          int stop;
          MPI_Recv(&stop, 1, MPI_INT, MASTER,
                   MSG_STOP, MPI_COMM_WORLD, &status);
          if (stop)
            {
              // MPI_Finalize();
              return 0;
            }

          // Receive data from master :
          int data[csize];
          MPI_Recv(data, csize, MPI_INT, MASTER, MSG_DATA,
                   MPI_COMM_WORLD, &status);

          // Reorder data in a picture :
          ImageSet<float> trans(NB_FILTERS);
          for (int i = 0; i < NB_FILTERS; i++)
            {
              trans[i] = Image<float>(n, n, NO_INIT);
              trans[i].setVal(0, data[0]);
              int range = 1;
              int current = 1;
              for (int j = 0; j < NCOEFFS; j++)
                {
                  for (int k = 0; k < 3; k++)
                    {
                      int si = range;
                      int sj = range;
                      if (k == 0)
                        sj = 0;
                      if (k == 1)
                        si = 0;
                      for (int ii = si; ii < range + si; ii++)
                        for (int jj = sj; jj < range + sj; jj++)
                          {
                            trans[i].setVal(ii, jj, data[current]);
                            current++;
                          }
                    }
                  range <<= 1;
                }
            }

          // Convert data into filters :
          ImageSet<float> filter(NB_FILTERS);
          for (int i = 0; i < NB_FILTERS; i++)
            filter[i] = matrixMult(transpose(hmat),
                                   matrixMult(trans[i], hmat));

          // Compute the dot product of the filters :
          int nb_prod = (NB_FILTERS * (NB_FILTERS - 1)) / 2;
          ImageSet<float> prod(nb_prod);
          int k = 0;
          for (int j = 0; j < NB_FILTERS - 1; j++)
            for (int i = j + 1; i < NB_FILTERS; i++)
              {
                prod[k] = filter[j] * filter[i];
                k++;
              }
          float dotprod = 0.0f;
          for (int i = 0; i < nb_prod; i++)
            dotprod += sum(prod[i]);

          // Instantiate a ModelManager :
          ModelManager manager("Open Attention Model");

          // Instantiate our various ModelComponents :
          nub::soft_ref<RawVisualCortex> vcx(new RawVisualCortex(manager));
          manager.addSubComponent(vcx);
          manager.setOptionValString(&OPT_MaxNormType, "Maxnorm");

          for (int i = 0; i < NB_FILTERS; i++)
            {
              // Create a channel attached to each filter :
              nub::soft_ref<ConvolveChannel>
                channel(new ConvolveChannel(manager));
              char txt[100];
              sprintf(txt, "Convolve%d", i);
              channel->setDescriptiveName(txt);
              sprintf(txt, "conv%d", i);
              channel->setTagName(txt);

              // Assign the filter to the channel:
              channel->setFilter(filter[i], CONV_BOUNDARY_ZERO);

              // Attach the channel to our visual cortex:
              vcx->addSubChan(channel);
            }

          // Let's get all our ModelComponent instances started :
          manager.exportOptions(MC_RECURSE);
          manager.setModelParamString("UsingFPE", "false");
          manager.setModelParamString("TestMode", "true");

          // Reduce amount of log messages :
          MYLOGVERB = loglev;

          // Start the manager :
          manager.start();

          float error = 0.0f;

          // Compute error for each picture :
          for (int pic = 0; pic < NB_PICS; pic++)
            {
              // Process the image through the visual cortex :
              vcx->input(InputFrame::fromRgb(&input[pic]));

              // Get the resulting saliency map :
              Image<float> sm = vcx->getOutput();

              // Reset the visual cortex :
              vcx->reset(MC_RECURSE);

              // Normalize the saliency map :
              inplaceNormalize(sm, 0.0f, 255.0f);

              // Get the average saliency :
              double avgsal = mean(sm);

              // Get the map level to scale things down :
              const LevelSpec lspec =
                vcx->getModelParamVal<LevelSpec>("LevelSpec");
              int sml = lspec.mapLevel();

              // Scale the radius :
              int radius = RAD >> sml;

              // Get the saliency on each end of saccade :
              float chsal = 0;
              int sc = 1 << sml;
              for (int i = 0; i < nch[pic]; i++)
                chsal += getLocalMax(sm, ch[i][pic] / sc, radius);
              chsal /= nch[pic];
              float kpsal = 0;
              for (int i = 0; i < nkp[pic]; i++)
                kpsal += getLocalMax(sm, kp[i][pic] / sc, radius);
              kpsal /= nkp[pic];
              float pwsal = 0;
              for (int i = 0; i < npw[pic]; i++)
                pwsal += getLocalMax(sm, pw[i][pic] / sc, radius);
              pwsal /= npw[pic];
              float wkmsal = 0;
              for (int i = 0; i < nwkm[pic]; i++)
                wkmsal += getLocalMax(sm, wkm[i][pic] / sc, radius);
              wkmsal /= nwkm[pic];

              float goodsal = (chsal + kpsal + pwsal + wkmsal) / 4;

              // Compute the error :
              error += (1 + avgsal) / (1 + goodsal);
            }

          // Stop all our ModelComponents :
          manager.stop();

          // Compute total result :
          float result = (error / NB_PICS) + 0.00001f * fabs(dotprod);
          result = 1.0f / result;

          // Send the result to the master :
          MPI_Send(&result, 1, MPI_FLOAT, MASTER, MSG_RESULT,
                   MPI_COMM_WORLD);
        }
    }

#endif // HAVE_MPI_H

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
