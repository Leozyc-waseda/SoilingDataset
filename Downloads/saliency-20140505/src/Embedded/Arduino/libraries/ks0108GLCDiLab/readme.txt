Installation
--------------------------------------------------------------------------------

To install this library, just place this entire folder as a subfolder in your
Arduino hardware/libraries folder.

When installed, this library should look like:

Arduino/hardware/libraries/ks0108              (this library's folder)
Arduino/hardware/libraries/ks0108/ks0108.cpp   (the library implementation file)
Arduino/hardware/libraries/ks0108/ks0108.h     (the library header file)
Arduino/hardware/libraries/ks0108/Arial14.h    (the definition for 14 point Arial Font)
Arduino/hardware/libraries/ks0108/keywords.txt (the syntax coloring file)
Arduino/hardware/libraries/ks0108/examples     (diectory containing the example test sketch)
Arduino/hardware/libraries/ks0108/readme.txt   (this file)

Building
--------------------------------------------------------------------------------

After this library is installed, you just have to start the Arduino application.

To use this library in a sketch, go to the Sketch | Import Library menu and
select ks0108.  This will add a corresponding line to the top of your sketch:
#include <ks0108.h>. It will also add lines for all the font definitions you have
in the ks0108 library directory. You should remove the includes for any fonts you
don't use in a sketch, they use a lot of program memory.

To stop using this library, delete that line and any included fonts from your sketch.

After a successful build of this library, a new file named 'ks0108.o' will appear
in ks0108 library directory. If you want to make any changes to the ks0108 library
you must delete ks0108.o before recompiling your sketch for the changes to be
recognized by your sketch. The new "Test.o" with your code will appear after the next
verify or compile (If there are no syntax errors in the changed code).

