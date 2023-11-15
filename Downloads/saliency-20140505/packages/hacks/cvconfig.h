/* cvconfig.h.  Generated from cvconfig.h.in by configure.  */
/* cvconfig.h.in.  Generated from configure.in by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those systems.
   */
/* #undef CRAY_STACKSEG_END */

/* Define to 1 if using `alloca.c'. */
/* #undef C_ALLOCA */

/* Define to 1 if you have `alloca', as a function or macro. */
#define HAVE_ALLOCA 1

/* Define to 1 if you have <alloca.h> and it should be used (not on Ultrix).
   */
#define HAVE_ALLOCA_H 1

/* Define to 1 if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* V4L capturing support */
#define HAVE_CAMV4L /**/

/* V4L2 capturing support */
#define HAVE_CAMV4L2 /**/

/* Carbon windowing environment */
/* #undef HAVE_CARBON */

/* IEEE1394 capturing support */
#define HAVE_DC1394 /**/

/* libdc1394 0.9.4 or 0.9.5 */
/* #undef HAVE_DC1394_095 */

/* IEEE1394 capturing support */
/* #undef HAVE_DC1394_2 */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* FFMPEG video library */
#define HAVE_FFMPEG /**/

/* Define to 1 if you have the <ffmpeg/avcodec.h> header file. */
/* #undef HAVE_FFMPEG_AVCODEC_H */

/* Define to 1 if you have the <ffmpeg/avformat.h> header file. */
/* #undef HAVE_FFMPEG_AVFORMAT_H */

/* FFMPEG helper library */
#define HAVE_FFMPEG_SWSCALE /**/

/* Define to 1 if you have the <ffmpeg/swscale.h> header file. */
/* #undef HAVE_FFMPEG_SWSCALE_H */

/* GStreamer multimedia framework */
/* #undef HAVE_GSTREAMER */

/* GTK+ 2.0 Thread support */
#define HAVE_GTHREAD /**/

/* GTK+ 2.x toolkit */
#define HAVE_GTK /**/

/* OpenEXR codec */
/* #undef HAVE_ILMIMF */

/* Apple ImageIO Framework */
/* #undef HAVE_IMAGEIO */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* JPEG-2000 codec */
#define HAVE_JASPER /**/

/* IJG JPEG codec */
#define HAVE_JPEG /**/

/* Define to 1 if you have the <libavcodec/avcodec.h> header file. */
#define HAVE_LIBAVCODEC_AVCODEC_H 1

/* Define to 1 if you have the <libavformat/avformat.h> header file. */
#define HAVE_LIBAVFORMAT_AVFORMAT_H 1

/* Define to 1 if you have the `dl' library (-ldl). */
#define HAVE_LIBDL 1

/* Define to 1 if you have the `m' library (-lm). */
#define HAVE_LIBM 1

/* libpng/png.h needs to be included */
/* #undef HAVE_LIBPNG_PNG_H */

/* Define to 1 if you have the `pthread' library (-lpthread). */
#define HAVE_LIBPTHREAD 1

/* Define to 1 if you have the <libswscale/swscale.h> header file. */
#define HAVE_LIBSWSCALE_SWSCALE_H 1

/* Define to 1 if you have the `z' library (-lz). */
#define HAVE_LIBZ 1

/* Define to 1 if you have the `lrint' function. */
#define HAVE_LRINT 1

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Support mmx instructions */
#define HAVE_MMX /**/

/* PNG codec */
#define HAVE_PNG /**/

/* Define to 1 if you have the `png_get_valid' function. */
#define HAVE_PNG_GET_VALID 1

/* png.h needs to be included */
#define HAVE_PNG_H /**/

/* Define to 1 if you have the `png_set_tRNS_to_alpha' function. */
#define HAVE_PNG_SET_TRNS_TO_ALPHA 1

/* QuickTime video libraries */
/* #undef HAVE_QUICKTIME */

/* Support SSE (Streaming SIMD Extensions) instructions */
#define HAVE_SSE /**/

/* Support SSE2 (Streaming SIMD Extensions 2) instructions */
#define HAVE_SSE2 /**/

/* Support SSE3 (Streaming SIMD Extensions 3) instructions */
#define HAVE_SSE3 /**/

/* Support SSSE3 (Supplemental Streaming SIMD Extensions 3) instructions */
/* #undef HAVE_SSSE3 */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* TIFF codec */
#define HAVE_TIFF /**/

/* Unicap video capture library */
/* #undef HAVE_UNICAP */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* BLAS and LAPACK from Accelerate/vecLib framework */
/* #undef HAVE_VECLIB */

/* Xine video library */
/* #undef HAVE_XINE */

/* LZ77 compression/decompression library (used for PNG) */
#define HAVE_ZLIB /**/

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "opencv"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "opencvlibrary-devel@lists.sourceforge.net"

/* Define to the full name of this package. */
#define PACKAGE_NAME "opencv"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "opencv 2.0.0"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "opencv"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.0.0"

/* If using the C implementation of alloca, define if you know the
   direction of stack growth for your system; otherwise it will be
   automatically deduced at runtime.
	STACK_DIRECTION > 0 => grows toward higher addresses
	STACK_DIRECTION < 0 => grows toward lower addresses
	STACK_DIRECTION = 0 => direction of growth unknown */
/* #undef STACK_DIRECTION */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "2.0.0"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif
