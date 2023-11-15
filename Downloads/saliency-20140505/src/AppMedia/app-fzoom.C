/*
 *              Filtered Image Rescaling
 *
 *                by Dale Schumacher
 *
 */

/*
        Additional changes by Ray Gardener, Daylon Graphics Ltd.
        December 4, 1999

        Additional changes by Rob Peters, University of Southern California
        August 23, 2007

        Summary:

                - Horizontal filter contributions are calculated on the fly,
                  as each column is mapped from src to dst image. This lets
                  us omit having to allocate a temporary full horizontal stretch
                  of the src image.

                - If none of the src pixels within a sampling region differ,
                  then the output pixel is forced to equal (any of) the source pixel.
                  This ensures that filters do not corrupt areas of constant color.

                - Filter weight contribution results, after summing, are
                  rounded to the nearest pixel color value instead of
                  being casted to T (usually an int or char). Otherwise,
                  artifacting occurs.
*/

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-fzoom.C $
// $Id: app-fzoom.C 8722 2007-08-25 00:42:17Z rjpeters $

#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Raster/Raster.H"
#include "Util/Assert.H"
#include "Util/CpuTimer.H"
#include "Util/StringUtil.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/sformat.H"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h> // for getopt() et al
#include <vector>


/*
 *      command line interface
 */

static
void
usage(const char* argv0)
{
        fprintf(stderr, "usage: %s [-options] input output\n", argv0);
        fprintf(stderr, "\
options:\n\
        -x xsize                output x size\n\
        -y ysize                output y size\n\
        -f filter               filter type\n\
{b=box, t=triangle, q=bell, B=B-spline, h=hermite, l=Lanczos3, m=Mitchell}\n\
        input, output files to read/write.\n\
");
        exit(1);
}

int
main(int argc, char* argv[])
{
        int c;
        int xsize = 0, ysize = 0;

        RescaleType ftype = RESCALE_FILTER_TRIANGLE;

        while ((c = getopt(argc, argv, "x:y:f:")) != EOF) {
                switch(c) {
                case 'x': xsize = atoi(optarg); break;
                case 'y': ysize = atoi(optarg); break;
                case 'f': ftype = getRescaleTypeFromChar(*optarg); break;
                case '?': usage(argv[0]);
                default:  usage(argv[0]);
                }
        }

        if ((argc - optind) != 2) usage(argv[0]);
        const char* srcfile = argv[optind];
        const char* dstfile = argv[optind + 1];

        const Image<byte> src = Raster::ReadGray(srcfile);

        const Dims newdims(xsize <= 0 ? src.getWidth() : xsize,
                           ysize <= 0 ? src.getHeight() : ysize);

        CpuTimer tm;

        const Image<byte> dst = rescale(src, newdims, ftype);

        tm.mark();
        tm.report(sformat("%8s %dx%d -> %dx%d",
                          convertToString(ftype).c_str(),
                          src.getWidth(), src.getHeight(),
                          newdims.w(), newdims.h()).c_str());

        Raster::WriteGray(dst, dstfile);

        return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */
