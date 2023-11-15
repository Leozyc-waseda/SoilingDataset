/*!@file GA/GAChromosome.H A chromosome class for genetic algorithm. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/GA/GAChromosome.H $
// $Id: GAChromosome.H 6003 2005-11-29 17:22:45Z rjpeters $
//

#ifndef GACHROMOSOME_H
#define GACHROMOSOME_H

#include <iosfwd>

//! Chromosome class for genetic algorithm
/*! This is a chromosome class for genetic algorithm, which is an
    array of genes (int with value of -1, 0 or 1). */

class GAChromosome
{
  friend std::istream& operator>>(std::istream& in, GAChromosome& c);

  friend std::ostream& operator<<(std::ostream& in, GAChromosome& c);

public:

  // ############################################################
  /*! @name Constructor, destructor and initialization.*/
  //@{

  //! Uninitialized constructor. Need to call resize() later.
  GAChromosome();

  //! Construct from C array.
  GAChromosome(const int N, const int *a);

  //! Construct random chromosome.
  GAChromosome(const int N);

  //! Copy constructor.
  GAChromosome(const GAChromosome& c);

  //! Change size, do not initialize.
  void resize(const int N);

  //! Initialize from C array.
  void init(const int N, const int *a);

  //! Initialize randomly.
  void init(const int N);

  //! Destructor.
  ~GAChromosome();

  //@}

  // ############################################################
  /*! @name Acces functions */
  //@{

  //! Return the size of the chromosome.
  int get_size() const;

  //! Set the gene #i to a.
  void set_gene(const int i, const int a);

  //! Return the gene #i
  int get_gene(const int i) const;

  //! Set the fitness to a.
  void set_fitness(const float a);

  //! Return the fitness.
  float get_fitness() const;

  //! Set the linear fitness to a.
  void set_linear_fitness(const float a);

  //! Return the linear fitness.
  float get_linear_fitness() const;

  //! Set breedings to a.
  void set_breedings(const int a);

  //! Return the breedings.
  int get_breedings() const;

  //@}

  // ############################################################
  /*! @name Operators overloading */
  //@{

  //! Overload of assignment operator.
  GAChromosome& operator=(const GAChromosome& c);

  //! Overload of less-than operator.
  /*! c1 < c2 if c1 as less breedings than c2 */
  bool operator<(const GAChromosome& c) const;

  //@}

  // ############################################################
  /*! @name Evolve functions */
  //@{

  //! Change one gene randomly.
  void mutation();

  //! Add one breeding.
  void add_breeding();

  //! Use one breeding.
  void use_breeding();

  //@}

private:
  float fitness;
  float linear_fitness;
  int breedings;
  int size;
  int *genes;
};

//! << overloading.
std::istream& operator>> (std::istream& in, GAChromosome& c);

//! >> overloading.
std::ostream& operator<< (std::ostream& out, GAChromosome& c);

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
