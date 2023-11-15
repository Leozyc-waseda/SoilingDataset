
#ifndef CUDAPYRBUILDER_H_DEFINED
#define CUDAPYRBUILDER_H_DEFINED

#include "CUDA/CudaConvolutions.H"
#include "CUDA/CudaImage.H"
#include "Image/PyramidTypes.H"

#include <deque>

template <class T> class CudaImageSet;
template <class T> class PixRGB;
template <class T> class CudaPyramidCache;

// ######################################################################
//! An interface class for creating dyadic pyramids from input images.
/*! A dyadic image pyramid is an array of images such that the next image
    is obtained from the previous one by applying some filter and
    decimating the image by a factor 2 horizontally and vertically.

    The PyrBuilder classes essentially wrap one of the buildPyr*()
    functions, binding preset values to all of the trailing arguments. This
    allows all of the buildPyr*() functions to be presented behind a
    uniform interface.

    PyrBuilder defers the actual handling of input images through the
    abstract function build(). Different subclasses may implement this to
    apply different filters (e.g., gaussian, laplacian, gabor) before
    decimation. GenericPyrBuilder offers a generic implementation, while
    GaussianPyrBuilder, LaplacianPyrBuilder, and OrientedPyrBuilder are
    specialized for single filter types.
*/
template <class T>
class CudaPyrBuilder
{
public:
  //! Create an empty CudaPyrBuilder
  CudaPyrBuilder();

  // default copy contructor and assignment are okay

  //! Virtual destructor for safe inheritance.
  virtual ~CudaPyrBuilder();

  //! Cloning constructor useful to make clones from pointers:
  virtual CudaPyrBuilder<T>* clone() const = 0;

  //! Create a pyramid from an input image, with given depth (=nb levels)
  /*! @param firstlevel if non-zero, then pyramid levels prior to
      firstlevel may be skipped and left empty, as long as the
      remaining pyramid levels contain the same values that they would
      normally contain -- this parameter is only an optimization hint,
      and subclasses are free to ignore it if they so choose

      @param depth number of levels the pyramid should contain (note
      that if firstlevel is non-zero, then some of those levels may be
      empty)
   */
  virtual CudaImageSet<T> build(const CudaImage<T>& image,
                            const int firstlevel, const int depth,
                            CudaPyramidCache<T>* cache = 0) = 0;

  //! reset the pyramid - whatever that means for a specific pyramid
  /*! a no op implementation is given in this base class - so if pyramids
    don't need this, they don't have to worry about it */
  virtual void reset();

private:
};


// ######################################################################
//! This class implements Reichardt motion detections
/*! The class uses the filter results of pyramids of other types
  to implement Reichardt motion detection. The pyramids of the previous
  time step are stored and used with the pyramid of this time step and
  a version of the current pyramid, in which each level is shifted by the
  same amount dx and dy. The results are stored in the Reichardt pyramid and
  represent motion at different speeds and spatial resolutions. */
template <class T>
class CudaReichardtPyrBuilder : public CudaPyrBuilder<T>
{
public:
  CudaReichardtPyrBuilder();
  // ######################################################################
  // #### Constructors
  // ######################################################################
  //! build with depth, dx, dy and type, but no input image yet
  /*! @param dx the number of pixels to shift into the x-direction
    (can be fractional value)
    @param dy the number of pixels to shift into the y-direction
    @param typ the type of pyramid to use for the filtering
    @param gabor_theta orientation of a gabor filter (if typ is GaborX)
    @param intens used for gabor filter ... */
  CudaReichardtPyrBuilder(const float dx,
                      const float dy,
                      const PyramidType typ,
                      const float gabor_theta = 0.0F,
                      const float intens = 10.0F);

  // default copy constructor and assignment are okay
  // std::vector and std::deque handle the copying of the indivisual images

  //! cloning constructor
  virtual CudaReichardtPyrBuilder<T>* clone() const;

  //! Builds a Reichardt pyramid based on a image
  virtual CudaImageSet<T> build(const CudaImage<T>& image,
                            const int firstlevel, const int depth,
                            CudaPyramidCache<T>* cache = 0);

  //! reset the stored queues - overrides PyrBuilder<T>::reset()
  virtual void reset();

private:
  float itsDX, itsDY;
  PyramidType itsPtype;     // Gaussian, Laplacian, Gabor, etc.
  float itsGaborAngle;      // orientation tuning for Gabor pyr
  float itsGaborIntens;     // filter strength for Gabor pyr
  std::deque< CudaImageSet<T> > shifted, unshifted;
};

#endif
