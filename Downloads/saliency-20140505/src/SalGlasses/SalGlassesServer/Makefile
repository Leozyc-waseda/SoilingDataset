#############################################################################
#
#	Makefile for building the Glasses Server
#
#############################################################################

ifeq ($(DEBUG),y)
	CFLAGS += -O -g		# -O is need to expand inlines
else
	CFLAGS += -O3 -funroll-loops
endif

TARGET	= SalGlassesServer

PWD     := $(shell pwd)

GUMSTIX_BUILDROOT   = /home/elazary/gumstix/gumstix-buildroot
BUILD_ARM           = $(wildcard $(GUMSTIX_BUILDROOT)/build_arm*)
CROSS_COMPILE       = $(patsubst %gcc, %, $(wildcard $(BUILD_ARM)/staging_dir/bin/arm-linux-uclibc*-gcc))

ifeq ($(strip $(CROSS_COMPILE)),)
$(error Unable to detect Cross Compiler)
endif

#TARGET_ARCH=-Os -march=armv5te -mtune=iwmmxt -Wa,-mcpu=iwmmxt
TARGET_ARCH= -march=armv5te -mtune=iwmmxt -Wa,-mcpu=iwmmxt
CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++

#
# If you need additional serch paths for include files, then use the -I flag
# and add them to the CPPFLAGS variable
#

CPPFLAGS += -I./ -I../ -I/home/elazary/gumstix/corba/omniORB-4.1.0/include

CPPFLAGS += -DINT_IS_32_BITS=1 -DLONG_IS_32_BITS=1 -std=gnu99

#
# If you need addional search paths for library files, then use the -L flag
# and add them to LDFLAGS.
#

#LDFLAGS += ../Envision/envision.a

#
# If you need additional libraries, then use -lxxx to search for libxxx.a
#

#LDLIBS += /home/elazary/gumstix/corba/omniORB-4.1.0/lib/libomniORB4.a /home/elazary/gumstix/corba/omniORB-4.1.0/lib/libomnithread.a -lpthread
LDLIBS += -L/home/elazary/gumstix/corba/omniORB-4.1.0/lib -lomniORB4 -lomnithread -lpthread ../Envision/envision.a

LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)

.PHONY: all

all:	botControlIdl depend $(TARGET) install


#
# You can change the $(TARGET).c if that's not your main file and you can
# add additional .o files to the end of the line
#

SRCS	= $(TARGET).cc corbaUtil.cc SalGlassesSK.cc ImageOrbSK.cc capture.cc serial.cc

OBJS	= $(SRCS:.cc=.o)
		     
$(TARGET) : $(OBJS)

clean:
	rm -rf $(OBJS) core .depend $(TARGET)

depend .depend dep:
	$(CXX) $(CFLAGS) $(CPPFLAGS) -M $(SRCS) > .depend

botControlIdl:
	/home/elazary/gumstix/corba/omniORB-4.1.0/bin/omniidl -C. -bcxx ../../Corba/ImageOrb.idl
	/home/elazary/gumstix/corba/omniORB-4.1.0/bin/omniidl -C. -I../../ -bcxx ../../Corba/Objects/SalGlasses.idl

install:
	scp SalGlassesServer 192.168.1.101:

ifeq (.depend,$(wildcard .depend))
include .depend
endif

