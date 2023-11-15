/*!@file GA/GAPopulation.H A population class for genetic algorithm. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/GA/GAPopulation.H $
// $Id: GAPopulation.H 6003 2005-11-29 17:22:45Z rjpeters $
//

#ifndef GAPOPULTAION_H
#define GAPOPULTAION_H

#include "GA/GAChromosome.H"

#include <iosfwd>

//! Population class for genetic algorithm
/*! This is a population class for genetic algorithm. It's a size N
    array of chromosomes, and their potential offspring and comes with
    various method to handle them. */

class GAPopulation
{
  friend std::istream& operator>> (std::istream& in, GAPopulation& pop);

  friend std::ostream& operator<< (std::ostream& out, GAPopulation& pop);

public:

  // ############################################################
  /*! @name Constructor and destructor */
  //@{

  //! Uninitialized constructor. Need to call resize() later.
  GAPopulation();

  //! Construct a size N-population of random chromosomes of size a.
  GAPopulation(const int N, const int a);

  //! Change sizes, do not initialize.
  void resize(const int N, const int a);

  //! Initialize with N random chromosomes of size a.
  void init(const int N, const int a);

  //! Destructor
  ~GAPopulation();

  //@}

  // ############################################################
  /*! @name Access functions */
  //@{

  //! Set the chromosome #i to c.
  void set_chromosome(const int i, const GAChromosome& c);

  //! Return the chromosome #i.
  GAChromosome get_chromosome(const int i) const;

  //! Return the mean fitness.
  float get_mean_fitness() const;

  //! Return the standard deviation.
  float get_sigma() const;

  //@}

  // ############################################################
  /*! @name Evolution functions */
  //@{

  //! The offspring become the new generation.
  void update();

  //! Compute the mean and total fitness of the chromosomes.
  void compute_pop_fitness();

  //! Compute the standard deviation of the chromosomes' fitness.
  void compute_sigma();

  //! Apply a linear scaling to the chromosomes' fitness.
  void linear_scaling();

  //! Select chromosomes for breeding.
  void selection();

  //! Breed chromosomes to produce the offspring.
  void crossover();

  //! Mutate the offspring with probability 1/size.
  void mutate();

  //@}

private:
  int psize;
  int csize;
  float total_fitness;
  float total_linear_fitness;
  float mean_fitness;
  float sigma;
  GAChromosome *chromosomes;
  GAChromosome *offspring;
};

//! << overloading.
std::istream& operator<< (std::istream& in, GAPopulation& pop);

//! >> overloading.
std::ostream& operator>> (std::ostream& out, GAPopulation& pop);

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
