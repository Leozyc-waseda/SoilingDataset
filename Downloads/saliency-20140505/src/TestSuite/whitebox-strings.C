/*!@file TestSuite/whitebox-strings.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-strings.C $
// $Id: whitebox-strings.C 14814 2011-06-06 08:15:25Z kai $
//

#ifndef WHITEBOX_STRINGS_C_DEFINED
#define WHITEBOX_STRINGS_C_DEFINED

#include "TestSuite/TestSuite.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Util/sformat.H"
#include "Util/SortUtil.H"

#include <iterator> // for back_inserter()
#include <string>
#include <vector>

static void sformat_xx_format_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(sformat(0), "");
  REQUIRE_EQ(sformat("%s", ""), "");
}

static void sformat_xx_format_xx_2(TestSuite& suite)
{
  REQUIRE_EQ(sformat("%%"), "%");
  REQUIRE_EQ(sformat("%s", "hello"), "hello");
  REQUIRE_EQ(sformat("%d", 123), "123");
  REQUIRE_EQ(sformat("%4d", 123), " 123");
  REQUIRE_EQ(sformat("%04d", 123), "0123");
  REQUIRE_EQ(sformat("%-8s", "foobar"), "foobar  ");
  REQUIRE_EQ(sformat("%8s", "foobar"), "  foobar");
  REQUIRE_EQ(sformat("%6.2f", 10.55), " 10.55");
  REQUIRE_EQ(sformat("%s-frame%06d.%s", "mycool", 4567, "png"),
             "mycool-frame004567.png");
}

static void stringutil_xx_split_xx_1(TestSuite& suite)
{
  const std::string s = "";
  std::vector<std::string> toks;
  split(s, "", std::back_inserter(toks));

  REQUIRE_EQ(toks.size(), size_t(0));
}

static void stringutil_xx_split_xx_2(TestSuite& suite)
{
  const std::string s = "";
  std::vector<std::string> toks;
  split(s, "abcdefg", std::back_inserter(toks));

  REQUIRE_EQ(toks.size(), size_t(0));
}

static void stringutil_xx_split_xx_3(TestSuite& suite)
{
  const std::string s = "/:/";
  std::vector<std::string> toks;
  split(s, ":/", std::back_inserter(toks));

  REQUIRE_EQ(toks.size(), size_t(0));
}

static void stringutil_xx_split_xx_4(TestSuite& suite)
{
  const std::string s = "foo bar baz";
  std::vector<std::string> toks;
  split(s, ":/", std::back_inserter(toks));

  if (REQUIRE_EQ(toks.size(), size_t(1)))
    {
      REQUIRE_EQ(toks[0], "foo bar baz");
    }
}

static void stringutil_xx_split_xx_5(TestSuite& suite)
{
  const std::string s = "foo bar baz";
  std::vector<std::string> toks;
  split(s, "", std::back_inserter(toks));

  if (REQUIRE_EQ(toks.size(), size_t(1)))
    {
      REQUIRE_EQ(toks[0], "foo bar baz");
    }
}

static void stringutil_xx_split_xx_6(TestSuite& suite)
{
  const std::string s = "/:/foo :: bar:/baz//  /:";
  std::vector<std::string> toks;
  split(s, ":/", std::back_inserter(toks));

  if (REQUIRE_EQ(toks.size(), size_t(4)))
    {
      REQUIRE_EQ(toks[0], "foo ");
      REQUIRE_EQ(toks[1], " bar");
      REQUIRE_EQ(toks[2], "baz");
      REQUIRE_EQ(toks[3], "  ");
    }
}

static void stringutil_xx_join_xx_1(TestSuite& suite)
{
  const char* toks[] = { 0 };

  REQUIRE_EQ(join(&toks[0], &toks[0], ":::"), "");
}

static void stringutil_xx_join_xx_2(TestSuite& suite)
{
  const char* toks[] = { "foo", "bar", "baz" };

  REQUIRE_EQ(join(&toks[0], &toks[0] + 3, ""), "foobarbaz");
}

static void stringutil_xx_join_xx_3(TestSuite& suite)
{
  const char* toks[] = { "foo" };

  REQUIRE_EQ(join(&toks[0], &toks[0] + 1, ":::"), "foo");
}

static void stringutil_xx_join_xx_4(TestSuite& suite)
{
  const char* toks[] = { "foo", "bar", "baz" };

  REQUIRE_EQ(join(&toks[0], &toks[0] + 3, ":::"), "foo:::bar:::baz");
}

static void stringutil_xx_tolowercase_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(toLowerCase("abcdefghijklmnopqrstuvwxyz"),
             std::string("abcdefghijklmnopqrstuvwxyz"));

  REQUIRE_EQ(toLowerCase("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
             std::string("abcdefghijklmnopqrstuvwxyz"));

  REQUIRE_EQ(toLowerCase("01234567890`~!@#$%^&*()-_=+[]{}\\|;:'\",<.>/? \t\n"),
             std::string("01234567890`~!@#$%^&*()-_=+[]{}\\|;:'\",<.>/? \t\n"));

  REQUIRE_EQ(toLowerCase("SomE miXeD-Ca$e\nstr!NG!!?"),
             std::string("some mixed-ca$e\nstr!ng!!?"));
}

static void stringutil_xx_touppercase_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(toUpperCase("abcdefghijklmnopqrstuvwxyz"),
             std::string("ABCDEFGHIJKLMNOPQRSTUVWXYZ"));

  REQUIRE_EQ(toUpperCase("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
             std::string("ABCDEFGHIJKLMNOPQRSTUVWXYZ"));

  REQUIRE_EQ(toUpperCase("01234567890`~!@#$%^&*()-_=+[]{}\\|;:'\",<.>/? \t\n"),
             std::string("01234567890`~!@#$%^&*()-_=+[]{}\\|;:'\",<.>/? \t\n"));

  REQUIRE_EQ(toUpperCase("SomE miXeD-Ca$e\nstr!NG!!?"),
             std::string("SOME MIXED-CA$E\nSTR!NG!!?"));
}

static void stringutil_xx_levenshtein_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(levenshteinDistance("", ""), uint(0));
  REQUIRE_EQ(levenshteinDistance("", "a"), uint(1));
  REQUIRE_EQ(levenshteinDistance("a", ""), uint(1));
  REQUIRE_EQ(levenshteinDistance("a", "b"), uint(1));
  REQUIRE_EQ(levenshteinDistance("kitten", "kitten"), uint(0));
  REQUIRE_EQ(levenshteinDistance("kitten", "kittens"), uint(1));
  REQUIRE_EQ(levenshteinDistance("kittens", "kitten"), uint(1));
  REQUIRE_EQ(levenshteinDistance("kitten", "smitten"), uint(2));
  REQUIRE_EQ(levenshteinDistance("kitten", "kitchen"), uint(2));
  REQUIRE_EQ(levenshteinDistance("abcdefg", "12345678"), uint(8));

  REQUIRE_EQ(levenshteinDistance("12", "21"), uint(2));
  REQUIRE_EQ(damerauLevenshteinDistance("12", "21"), uint(1));
  REQUIRE_EQ(damerauLevenshteinDistance("123456", "124356"), uint(1));
  REQUIRE_EQ(damerauLevenshteinDistance("123456", "132546"), uint(2));
}

static void stringutil_xx_camelcase_xx_1(TestSuite& suite)
{
  std::set<std::string> acronyms;
  acronyms.insert("AGM");
  acronyms.insert("ALIAS");
  acronyms.insert("ASAC");
  acronyms.insert("EHC");
  acronyms.insert("FOA");
  acronyms.insert("FOV");
  acronyms.insert("FPE");
  acronyms.insert("FPS");
  acronyms.insert("H2SV1");
  acronyms.insert("IOR");
  acronyms.insert("ITC");
  acronyms.insert("MPEG");
  acronyms.insert("MRV");
  acronyms.insert("NP");
  acronyms.insert("SC");
  acronyms.insert("SDL");
  acronyms.insert("SM");
  acronyms.insert("SV");
  acronyms.insert("SVCOMP");
  acronyms.insert("SVEM");
  acronyms.insert("TCP");
  acronyms.insert("TRM");
  acronyms.insert("VODB");
  acronyms.insert("VB");
  acronyms.insert("VCO");
  acronyms.insert("VCC4");
  acronyms.insert("VCEM");
  acronyms.insert("VCX");
  acronyms.insert("WTA");

  // this set of words was compiled from the list of all
  // ModelOptionDef param names as of 2007-Mar-21, but feel free to
  // add other test case words here:
  const char* const words[] =
    {
      "AGMsaveResults"                    , "agm save results"                  ,
      "ALIAScamBttv"                      , "alias cam bttv"                    ,
      "ALIAScamMacbook"                   , "alias cam macbook"                 ,
      "ALIASeyeCompare"                   , "alias eye compare"                 ,
      "ALIASinitialVCO"                   , "alias initial vco"                 ,
      "ALIASmovie"                        , "alias movie"                       ,
      "ALIASmovieanim"                    , "alias movieanim"                   ,
      "ALIASmovieeyehead"                 , "alias movieeyehead"                ,
      "ALIASmoviefov"                     , "alias moviefov"                    ,
      "ALIASsaveChannelInternals"         , "alias save channel internals"      ,
      "ALIASsaveChannelOutputs"           , "alias save channel outputs"        ,
      "ALIASsaveall"                      , "alias saveall"                     ,
      "ALIASsaveall"                      , "alias saveall"                     ,
      "ALIASsurprise"                     , "alias surprise"                    ,
      "ALIAStop5"                         , "alias top5"                        ,
      "ASACconfigFile"                    , "asac config file"                  ,
      "ASACdrawBetaParts"                 , "asac draw beta parts"              ,
      "ASACdrawBiasParts"                 , "asac draw bias parts"              ,
      "ASACdrawDiffParts"                 , "asac draw diff parts"              ,
      "ASACdrawSeperableParts"            , "asac draw seperable parts"         ,
      "AsyncUi"                           , "async ui"                          ,
      "AsyncUiQueueSize"                  , "async ui queue size"               ,
      "AttentionGuidanceMapType"          , "attention guidance map type"       ,
      "AudioGrabberBits"                  , "audio grabber bits"                ,
      "AudioGrabberBufSamples"            , "audio grabber buf samples"         ,
      "AudioGrabberDevice"                , "audio grabber device"              ,
      "AudioGrabberFreq"                  , "audio grabber freq"                ,
      "AudioGrabberStereo"                , "audio grabber stereo"              ,
      "AudioMixerCdIn"                    , "audio mixer cd in"                 ,
      "AudioMixerDevice"                  , "audio mixer device"                ,
      "AudioMixerLineIn"                  , "audio mixer line in"               ,
      "AudioMixerMicIn"                   , "audio mixer mic in"                ,
      "BeowulfInitTimeout"                , "beowulf init timeout"              ,
      "BeowulfMaster"                     , "beowulf master"                    ,
      "BeowulfSelfDropLast"               , "beowulf self drop last"            ,
      "BeowulfSelfQlen"                   , "beowulf self qlen"                 ,
      "BeowulfSlaveNames"                 , "beowulf slave names"               ,
      "BiasFeatures"                      , "bias features"                     ,
      "BiasTRM"                           , "bias trm"                          ,
      "BlankBlink"                        , "blank blink"                       ,
      "BrainBoringDelay"                  , "brain boring delay"                ,
      "BrainBoringSMmv"                   , "brain boring sm mv"                ,
      "BrainSaveObjMask"                  , "brain save obj mask"               ,
      "BrainSaveWinnerFeatures"           , "brain save winner features"        ,
      "BrainTooManyShifts"                , "brain too many shifts"             ,
      "CacheSavePrefix"                   , "cache save prefix"                 ,
      "ChannelOutputRangeMax"             , "channel output range max"          ,
      "ChannelOutputRangeMin"             , "channel output range min"          ,
      "CheckPristine"                     , "check pristine"                    ,
      "ClipMaskFname"                     , "clip mask fname"                   ,
      "ColorComputeType"                  , "color compute type"                ,
      "CompColorDoubleOppWeight"          , "comp color double opp weight"      ,
      "CompColorSingleOppWeight"          , "comp color single opp weight"      ,
      "ComplexChannelSaveOutputMap"       , "complex channel save output map"   ,
      "DebugMode"                         , "debug mode"                        ,
      "DeinterlacerType"                  , "deinterlacer type"                 ,
      "DescriptorVecFOV"                  , "descriptor vec fov"                ,
      "DirectionChannelLowThresh"         , "direction channel low thresh"      ,
      "DirectionChannelTakeSqrt"          , "direction channel take sqrt"       ,
      "DiskDataStreamNumThreads"          , "disk data stream num threads"      ,
      "DiskDataStreamSavePath"            , "disk data stream save path"        ,
      "DiskDataStreamSavePeriod"          , "disk data stream save period"      ,
      "DiskDataStreamSleepUsecs"          , "disk data stream sleep usecs"      ,
      "DiskDataStreamUseMmap"             , "disk data stream use mmap"         ,
      "DoBottomUpContext"                 , "do bottom up context"              ,
      "DownVODBfname"                     , "down vodb fname"                   ,
      "DummyChannelFactor"                , "dummy channel factor"              ,
      "EFullImplementation"               , "e full implementation"             ,
      "EHCeyeTrackConfig"                 , "ehc eye track config"              ,
      "EsmInertiaHalfLife"                , "esm inertia half life"             ,
      "EsmInertiaRadius"                  , "esm inertia radius"                ,
      "EsmInertiaShiftThresh"             , "esm inertia shift thresh"          ,
      "EsmInertiaStrength"                , "esm inertia strength"              ,
      "EsmIorHalfLife"                    , "esm ior half life"                 ,
      "EsmIorRadius"                      , "esm ior radius"                    ,
      "EsmIorStrength"                    , "esm ior strength"                  ,
      "EvcColorSmoothing"                 , "evc color smoothing"               ,
      "EvcFlickerThresh"                  , "evc flicker thresh"                ,
      "EvcLevelSpec"                      , "evc level spec"                    ,
      "EvcMaxnormType"                    , "evc maxnorm type"                  ,
      "EvcMotionThresh"                   , "evc motion thresh"                 ,
      "EvcMultiScaleFlicker"              , "evc multi scale flicker"           ,
      "EvcNumDirections"                  , "evc num directions"                ,
      "EvcNumOrientations"                , "evc num orientations"              ,
      "EvcScaleBits"                      , "evc scale bits"                    ,
      "EvcType"                           , "evc type"                          ,
      "EventLogFileName"                  , "event log file name"               ,
      "EyeHeadControllerType"             , "eye head controller type"          ,
      "EyeSFileName"                      , "eye s file name"                   ,
      "EyeSNumSkip"                       , "eye s num skip"                    ,
      "EyeSPeriod"                        , "eye s period"                      ,
      "EyeTrackerEDFfname"                , "eye tracker e d ffname"            ,
      "EyeTrackerParDev"                  , "eye tracker par dev"               ,
      "EyeTrackerParTrig"                 , "eye tracker par trig"              ,
      "EyeTrackerSerDev"                  , "eye tracker ser dev"               ,
      "EyeTrackerType"                    , "eye tracker type"                  ,
      "FOAradius"                         , "foa radius"                        ,
      "FlushUiQueue"                      , "flush ui queue"                    ,
      "FoveaRadius"                       , "fovea radius"                      ,
      "FoveateInput"                      , "foveate input"                     ,
      "FoveateInputDepth"                 , "foveate input depth"               ,
      "FpuPrecision"                      , "fpu precision"                     ,
      "FpuRoundingMode"                   , "fpu rounding mode"                 ,
      "FrameGrabberBrightness"            , "frame grabber brightness"          ,
      "FrameGrabberByteSwap"              , "frame grabber byte swap"           ,
      "FrameGrabberChannel"               , "frame grabber channel"             ,
      "FrameGrabberColour"                , "frame grabber colour"              ,
      "FrameGrabberContrast"              , "frame grabber contrast"            ,
      "FrameGrabberDevice"                , "frame grabber device"              ,
      "FrameGrabberDims"                  , "frame grabber dims"                ,
      "FrameGrabberExposure"              , "frame grabber exposure"            ,
      "FrameGrabberFPS"                   , "frame grabber fps"                 ,
      "FrameGrabberGain"                  , "frame grabber gain"                ,
      "FrameGrabberGamma"                 , "frame grabber gamma"               ,
      "FrameGrabberHue"                   , "frame grabber hue"                 ,
      "FrameGrabberMode"                  , "frame grabber mode"                ,
      "FrameGrabberNbuf"                  , "frame grabber nbuf"                ,
      "FrameGrabberSaturation"            , "frame grabber saturation"          ,
      "FrameGrabberSharpness"             , "frame grabber sharpness"           ,
      "FrameGrabberShutter"               , "frame grabber shutter"             ,
      "FrameGrabberStreaming"             , "frame grabber streaming"           ,
      "FrameGrabberSubChan"               , "frame grabber sub chan"            ,
      "FrameGrabberType"                  , "frame grabber type"                ,
      "FrameGrabberWhiteBalBU"            , "frame grabber white bal b u"       ,
      "FrameGrabberWhiteBalRV"            , "frame grabber white bal r v"       ,
      "FrameGrabberWhiteness"             , "frame grabber whiteness"           ,
      "FrontVODBfname"                    , "front vodb fname"                  ,
      "FxSaveIllustrations"               , "fx save illustrations"             ,
      "FxSaveRawMaps"                     , "fx save raw maps"                  ,
      "GaborChannelIntensity"             , "gabor channel intensity"           ,
      "GetSingleChannelStats"             , "get single channel stats"          ,
      "GetSingleChannelStatsFile"         , "get single channel stats file"     ,
      "GetSingleChannelStatsTag"          , "get single channel stats tag"      ,
      "GistEstimatorType"                 , "gist estimator type"               ,
      "HeadMarkerRadius"                  , "head marker radius"                ,
      "HueBandWidth"                      , "hue band width"                    ,
      "IORtype"                           , "ior type"                          ,
      "ITCAttentionObjRecog"              , "itc attention obj recog"           ,
      "ITCInferoTemporalType"             , "itc infero temporal type"          ,
      "ITCMatchObjects"                   , "itc match objects"                 ,
      "ITCMatchingAlgorithm"              , "itc matching algorithm"            ,
      "ITCObjectDatabaseFileName"         , "itc object database file name"     ,
      "ITCOutputFileName"                 , "itc output file name"              ,
      "ITCPromptUserTrainDB"              , "itc prompt user train d b"         ,
      "ITCRecognitionMinMatch"            , "itc recognition min match"         ,
      "ITCRecognitionMinMatchPercent"     , "itc recognition min match percent" ,
      "ITCSortKeys"                       , "itc sort keys"                     ,
      "ITCSortObjects"                    , "itc sort objects"                  ,
      "ITCSortObjectsBy"                  , "itc sort objects by"               ,
      "ITCTrainObjectDB"                  , "itc train object d b"              ,
      "ImageDisplayType"                  , "image display type"                ,
      "InputBufferSize"                   , "input buffer size"                 ,
      "InputEchoDest"                     , "input echo dest"                   ,
      "InputFOV"                          , "input fov"                         ,
      "InputFrameCrop"                    , "input frame crop"                  ,
      "InputFrameDims"                    , "input frame dims"                  ,
      "InputFrameRange"                   , "input frame range"                 ,
      "InputFrameSource"                  , "input frame source"                ,
      "InputFramingImageName"             , "input framing image name"          ,
      "InputFramingImagePos"              , "input framing image pos"           ,
      "InputMPEGStreamCodec"              , "input mpeg stream codec"           ,
      "InputMPEGStreamPreload"            , "input mpeg stream preload"         ,
      "InputMbariFrameRange"              , "input mbari frame range"           ,
      "InputMbariPreserveAspect"          , "input mbari preserve aspect"       ,
      "InputOutputComboSpec"              , "input output combo spec"           ,
      "InputPreserveAspect"               , "input preserve aspect"             ,
      "InputRasterFileFormat"             , "input raster file format"          ,
      "InputYuvDims"                      , "input yuv dims"                    ,
      "InputYuvDimsLoose"                 , "input yuv dims loose"              ,
      "IntChannelOutputRangeMax"          , "int channel output range max"      ,
      "IntChannelOutputRangeMin"          , "int channel output range min"      ,
      "IntChannelScaleBits"               , "int channel scale bits"            ,
      "IntDecodeType"                     , "int decode type"                   ,
      "IntMathLowPass5"                   , "int math low pass5"                ,
      "IntMathLowPass9"                   , "int math low pass9"                ,
      "IntensityBandWidth"                , "intensity band width"              ,
      "IscanSerialPortDevice"             , "iscan serial port device"          ,
      "JobServerNumThreads"               , "job server num threads"            ,
      "KeepGoing"                         , "keep going"                        ,
      "LFullImplementation"               , "l full implementation"             ,
      "LearnTRM"                          , "learn trm"                         ,
      "LearnUpdateTRM"                    , "learn update trm"                  ,
      "LevelSpec"                         , "level spec"                        ,
      "LoadConfigFile"                    , "load config file"                  ,
      "LogVerb"                           , "log verb"                          ,
      "LsqSvdThresholdFactor"             , "lsq svd threshold factor"          ,
      "LsqUseWeightsFile"                 , "lsq use weights file"              ,
      "MRVdisplayOutput"                  , "mrv display output"                ,
      "MRVdisplayResults"                 , "mrv display results"               ,
      "MRVloadEvents"                     , "mrv load events"                   ,
      "MRVloadProperties"                 , "mrv load properties"               ,
      "MRVmarkCandidate"                  , "mrv mark candidate"                ,
      "MRVmarkFOE"                        , "mrv mark f o e"                    ,
      "MRVmarkInteresting"                , "mrv mark interesting"              ,
      "MRVmarkPrediction"                 , "mrv mark prediction"               ,
      "MRVopacity"                        , "mrv opacity"                       ,
      "MRVrescaleDisplay"                 , "mrv rescale display"               ,
      "MRVsaveEventNums"                  , "mrv save event nums"               ,
      "MRVsaveEvents"                     , "mrv save events"                   ,
      "MRVsaveOutput"                     , "mrv save output"                   ,
      "MRVsavePositions"                  , "mrv save positions"                ,
      "MRVsaveProperties"                 , "mrv save properties"               ,
      "MRVsaveResults"                    , "mrv save results"                  ,
      "MRVshowEventLabels"                , "mrv show event labels"             ,
      "MRVsizeAvgCache"                   , "mrv size avg cache"                ,
      "MapLevel"                          , "map level"                         ,
      "MaxImageDisplay"                   , "max image display"                 ,
      "MaxNormType"                       , "max norm type"                     ,
      "MgzOutputStreamCompLevel"          , "mgz output stream comp level"      ,
      "MovieHertz"                        , "movie hertz"                       ,
      "MovingAvgFactor"                   , "moving avg factor"                 ,
      "MultiRetinaDepth"                  , "multi retina depth"                ,
      "NPconfig"                          , "np config"                         ,
      "NerdCamConfigFile"                 , "nerd cam config file"              ,
      "NumColorBands"                     , "num color bands"                   ,
      "NumDirections"                     , "num directions"                    ,
      "NumEOrients"                       , "num e orients"                     ,
      "NumIntensityBands"                 , "num intensity bands"               ,
      "NumLOrients"                       , "num l orients"                     ,
      "NumOrientations"                   , "num orientations"                  ,
      "NumSatBands"                       , "num sat bands"                     ,
      "NumSkipFrames"                     , "num skip frames"                   ,
      "NumTOrients"                       , "num t orients"                     ,
      "NumTestingFrames"                  , "num testing frames"                ,
      "NumTheta"                          , "num theta"                         ,
      "NumTrainingFrames"                 , "num training frames"               ,
      "NumXOrients"                       , "num x orients"                     ,
      "Nv2CompactDisplay"                 , "nv2 compact display"               ,
      "Nv2ExtendedDisplay"                , "nv2 extended display"              ,
      "Nv2Network"                        , "nv2 network"                       ,
      "OriInteraction"                    , "ori interaction"                   ,
      "OrientComputeType"                 , "orient compute type"               ,
      "OutputFrameDims"                   , "output frame dims"                 ,
      "OutputFrameRange"                  , "output frame range"                ,
      "OutputFrameSink"                   , "output frame sink"                 ,
      "OutputMPEGStreamBitRate"           , "output mpeg stream bit rate"       ,
      "OutputMPEGStreamBufSize"           , "output mpeg stream buf size"       ,
      "OutputMPEGStreamCodec"             , "output mpeg stream codec"          ,
      "OutputMPEGStreamFrameRate"         , "output mpeg stream frame rate"     ,
      "OutputMbariShowFrames"             , "output mbari show frames"          ,
      "OutputPreserveAspect"              , "output preserve aspect"            ,
      "OutputRasterFileFormat"            , "output raster file format"         ,
      "OutputZoom"                        , "output zoom"                       ,
      "PixelsPerDegree"                   , "pixels per degree"                 ,
      "Polyconfig"                        , "polyconfig"                        ,
      "ProfileOutFile"                    , "profile out file"                  ,
      "PsychoDisplayBackgroundColor"      , "psycho display background color"   ,
      "PsychoDisplayBlack"                , "psycho display black"              ,
      "PsychoDisplayFullscreen"           , "psycho display fullscreen"         ,
      "PsychoDisplayPriority"             , "psycho display priority"           ,
      "PsychoDisplayRefreshUsec"          , "psycho display refresh usec"       ,
      "PsychoDisplayTextColor"            , "psycho display text color"         ,
      "PsychoDisplayWhite"                , "psycho display white"              ,
      "QdisplayEcho"                      , "qdisplay echo"                     ,
      "QdisplayPrefDims"                  , "qdisplay pref dims"                ,
      "QdisplayPrefMaxDims"               , "qdisplay pref max dims"            ,
      "RawInpRectBorder"                  , "raw inp rect border"               ,
      "RetinaFlipHoriz"                   , "retina flip horiz"                 ,
      "RetinaFlipVertic"                  , "retina flip vertic"                ,
      "RetinaMaskFname"                   , "retina mask fname"                 ,
      "RetinaSaveInput"                   , "retina save input"                 ,
      "RetinaSaveOutput"                  , "retina save output"                ,
      "RetinaStdSavePyramid"              , "retina std save pyramid"           ,
      "RetinaType"                        , "retina type"                       ,
      "SCeyeBlinkDuration"                , "sc eye blink duration"             ,
      "SCeyeBlinkWaitTime"                , "sc eye blink wait time"            ,
      "SCeyeFrictionMu"                   , "sc eye friction mu"                ,
      "SCeyeInitialPosition"              , "sc eye initial position"           ,
      "SCeyeMaxIdleSecs"                  , "sc eye max idle secs"              ,
      "SCeyeMinSacLen"                    , "sc eye min sac len"                ,
      "SCeyeSpringK"                      , "sc eye spring k"                   ,
      "SCeyeThreshMaxCovert"              , "sc eye thresh max covert"          ,
      "SCeyeThreshMinNum"                 , "sc eye thresh min num"             ,
      "SCeyeThreshMinOvert"               , "sc eye thresh min overt"           ,
      "SCeyeThreshSalWeigh"               , "sc eye thresh sal weigh"           ,
      "SCheadFrictionMu"                  , "sc head friction mu"               ,
      "SCheadInitialPosition"             , "sc head initial position"          ,
      "SCheadMaxIdleSecs"                 , "sc head max idle secs"             ,
      "SCheadMinSacLen"                   , "sc head min sac len"               ,
      "SCheadSpringK"                     , "sc head spring k"                  ,
      "SCheadThreshMaxCovert"             , "sc head thresh max covert"         ,
      "SCheadThreshMinNum"                , "sc head thresh min num"            ,
      "SCheadThreshMinOvert"              , "sc head thresh min overt"          ,
      "SCheadThreshSalWeigh"              , "sc head thresh sal weigh"          ,
      "SDLdisplayDims"                    , "sdl display dims"                  ,
      "SDLdisplayFullscreen"              , "sdl display fullscreen"            ,
      "SDLdisplayPriority"                , "sdl display priority"              ,
      "SDLdisplayRefreshUsec"             , "sdl display refresh usec"          ,
      "SMTnumThreads"                     , "sm tnum threads"                   ,
      "SMfastInputCoeff"                  , "sm fast input coeff"               ,
      "SMfastItoVcoeff"                   , "sm fast ito vcoeff"                ,
      "SMginhDecay"                       , "sm ginh decay"                     ,
      "SMsaveCumResults"                  , "sm save cum results"               ,
      "SMsaveResults"                     , "sm save results"                   ,
      "SMuseBlinkSuppress"                , "sm use blink suppress"             ,
      "SMuseSacSuppress"                  , "sm use sac suppress"               ,
      "SVCOMPMultiRetinaDepth"            , "svcomp multi retina depth"         ,
      "SVCOMPcacheSize"                   , "svcomp cache size"                 ,
      "SVCOMPdisplayHumanEye"             , "svcomp display human eye"          ,
      "SVCOMPfoveaSCtype"                 , "svcomp fovea sc type"              ,
      "SVCOMPiframePeriod"                , "svcomp iframe period"              ,
      "SVCOMPnumFoveas"                   , "svcomp num foveas"                 ,
      "SVCOMPsaveEyeCombo"                , "svcomp save eye combo"             ,
      "SVCOMPsaveMegaCombo"               , "svcomp save mega combo"            ,
      "SVCOMPsaveXcombo"                  , "svcomp save xcombo"                ,
      "SVCOMPsaveXcomboOrig"              , "svcomp save xcombo orig"           ,
      "SVCOMPsaveYcombo"                  , "svcomp save ycombo"                ,
      "SVCOMPsaveYcomboOrig"              , "svcomp save ycombo orig"           ,
      "SVCOMPuseTRMmax"                   , "svcomp use trm max"                ,
      "SVEMbufDims"                       , "svem buf dims"                     ,
      "SVEMdelayCacheSize"                , "svem delay cache size"             ,
      "SVEMdisplaySacNum"                 , "svem display sac num"              ,
      "SVEMmaxCacheSize"                  , "svem max cache size"               ,
      "SVEMmaxComboWidth"                 , "svem max combo width"              ,
      "SVEMnumRandomSamples"              , "svem num random samples"           ,
      "SVEMoutFname"                      , "svem out fname"                    ,
      "SVEMsampleAtStart"                 , "svem sample at start"              ,
      "SVEMsaveMegaCombo"                 , "svem save mega combo"              ,
      "SVEMuseIOR"                        , "svem use ior"                      ,
      "SVcropFOA"                         , "sv crop foa"                       ,
      "SVdisplayAdditive"                 , "sv display additive"               ,
      "SVdisplayBoring"                   , "sv display boring"                 ,
      "SVdisplayEye"                      , "sv display eye"                    ,
      "SVdisplayEyeLinks"                 , "sv display eye links"              ,
      "SVdisplayFOA"                      , "sv display foa"                    ,
      "SVdisplayFOALinks"                 , "sv display foa links"              ,
      "SVdisplayHead"                     , "sv display head"                   ,
      "SVdisplayHeadLinks"                , "sv display head links"             ,
      "SVdisplayHighlights"               , "sv display highlights"             ,
      "SVdisplayInterp"                   , "sv display interp"                 ,
      "SVdisplayMapFactor"                , "sv display map factor"             ,
      "SVdisplayMapType"                  , "sv display map type"               ,
      "SVdisplayPatch"                    , "sv display patch"                  ,
      "SVdisplaySMmodulate"               , "sv display sm modulate"            ,
      "SVdisplayTime"                     , "sv display time"                   ,
      "SVeyeSimFname"                     , "sv eye sim fname"                  ,
      "SVeyeSimPeriod"                    , "sv eye sim period"                 ,
      "SVeyeSimTrash"                     , "sv eye sim trash"                  ,
      "SVfontSize"                        , "sv font size"                      ,
      "SVfoveateTraj"                     , "sv foveate traj"                   ,
      "SVmegaCombo"                       , "sv mega combo"                     ,
      "SVmotionSixPack"                   , "sv motion six pack"                ,
      "SVsaveTRMXcombo"                   , "sv save trm xcombo"                ,
      "SVsaveTRMYcombo"                   , "sv save trm ycombo"                ,
      "SVsaveTraj"                        , "sv save traj"                      ,
      "SVsaveXcombo"                      , "sv save xcombo"                    ,
      "SVsaveYcombo"                      , "sv save ycombo"                    ,
      "SVstatsFname"                      , "sv stats fname"                    ,
      "SVuseLargerDrawings"               , "sv use larger drawings"            ,
      "SVwarp3D"                          , "sv warp3 d"                        ,
      "SVxwindow"                         , "sv xwindow"                        ,
      "SaccadeControllerEyeType"          , "saccade controller eye type"       ,
      "SaccadeControllerHeadType"         , "saccade controller head type"      ,
      "SaliencyMapType"                   , "saliency map type"                 ,
      "SatBandWidth"                      , "sat band width"                    ,
      "SaveConfigFile"                    , "save config file"                  ,
      "SaveInputCopy"                     , "save input copy"                   ,
      "ScorrChannelRadius"                , "scorr channel radius"              ,
      "ShapeEstimatorMode"                , "shape estimator mode"              ,
      "ShapeEstimatorSmoothMethod"        , "shape estimator smooth method"     ,
      "ShapeEstimatorUseLargeNeigh"       , "shape estimator use large neigh"   ,
      "ShiftInputToEye"                   , "shift input to eye"                ,
      "ShiftInputToEyeBGcol"              , "shift input to eye b gcol"         ,
      "ShowHelpMessage"                   , "show help message"                 ,
      "ShowMemStats"                      , "show mem stats"                    ,
      "ShowMemStatsUnits"                 , "show mem stats units"              ,
      "ShowSvnVersion"                    , "show svn version"                  ,
      "ShowVersion"                       , "show version"                      ,
      "SimEventQueueType"                 , "sim event queue type"              ,
      "SimulationTimeStep"                , "simulation time step"              ,
      "SimulationTooMuchTime"             , "simulation too much time"          ,
      "SimulationViewerType"              , "simulation viewer type"            ,
      "SingleChannelBeoServerQuickMode"   , "single channel beo server quick mode",
      "SingleChannelQueueLen"             , "single channel queue len"          ,
      "SingleChannelSaveFeatureMaps"      , "single channel save feature maps"  ,
      "SingleChannelSaveOutputMap"        , "single channel save output map"    ,
      "SingleChannelSaveRawMaps"          , "single channel save raw maps"      ,
      "SingleChannelSurpriseLogged"       , "single channel surprise logged"    ,
      "SingleChannelSurpriseNeighUpdFac"  , "single channel surprise neigh upd fac",
      "SingleChannelSurpriseProbe"        , "single channel surprise probe"     ,
      "SingleChannelSurpriseSLfac"        , "single channel surprise s lfac"    ,
      "SingleChannelSurpriseSQlen"        , "single channel surprise s qlen"    ,
      "SingleChannelSurpriseSSfac"        , "single channel surprise s sfac"    ,
      "SingleChannelSurpriseUpdFac"       , "single channel surprise upd fac"   ,
      "SingleChannelTimeDecay"            , "single channel time decay"         ,
      "SingleChannelUseSplitCS"           , "single channel use split c s"      ,
      "SmfxNormType"                      , "smfx norm type"                    ,
      "SmfxRescale512"                    , "smfx rescale512"                   ,
      "SmfxVcType"                        , "smfx vc type"                      ,
      "SockServPort"                      , "sock serv port"                    ,
      "SubmapAlgoType"                    , "submap algo type"                  ,
      "TCPcommunicatorDisableShm"         , "tcp communicator disable shm"      ,
      "TCPcommunicatorIPaddr"             , "tcp communicator i paddr"          ,
      "TCPcommunicatorInDropLast"         , "tcp communicator in drop last"     ,
      "TCPcommunicatorInQlen"             , "tcp communicator in qlen"          ,
      "TCPcommunicatorOuDropLast"         , "tcp communicator ou drop last"     ,
      "TCPcommunicatorOuQlen"             , "tcp communicator ou qlen"          ,
      "TFullImplementation"               , "t full implementation"             ,
      "TRMKillStaticCoeff"                , "trm kill static coeff"             ,
      "TRMKillStaticThresh"               , "trm kill static thresh"            ,
      "TRMkillN"                          , "trm kill n"                        ,
      "TRMsaveResults"                    , "trm save results"                  ,
      "TargetMaskFname"                   , "target mask fname"                 ,
      "TaskRelevanceMapType"              , "task relevance map type"           ,
      "TcorrChannelFrameLag"              , "tcorr channel frame lag"           ,
      "TdcLocalMax"                       , "tdc local max"                     ,
      "TdcRectifyTd"                      , "tdc rectify td"                    ,
      "TdcSaveMaps"                       , "tdc save maps"                     ,
      "TdcSaveMapsNormalized"             , "tdc save maps normalized"          ,
      "TdcSaveRawData"                    , "tdc save raw data"                 ,
      "TdcSaveSumo"                       , "tdc save sumo"                     ,
      "TdcSaveSumo2"                      , "tdc save sumo2"                    ,
      "TdcTemporalMax"                    , "tdc temporal max"                  ,
      "TestMode"                          , "test mode"                         ,
      "TextLogFile"                       , "text log file"                     ,
      "TopdownContextSpec"                , "topdown context spec"              ,
      "TraceXEvents"                      , "trace x events"                    ,
      "TrainingSetDecimation"             , "training set decimation"           ,
      "TrainingSetRebalance"              , "training set rebalance"            ,
      "TrainingSetRebalanceGroupSize"     , "training set rebalance group size" ,
      "TrainingSetRebalanceThresh"        , "training set rebalance thresh"     ,
      "UcbMpegFrameRate"                  , "ucb mpeg frame rate"               ,
      "UcbMpegQuality"                    , "ucb mpeg quality"                  ,
      "UnderflowStrategy"                 , "underflow strategy"                ,
      "UpVODBfname"                       , "up vodb fname"                     ,
      "UseH2SV1"                          , "use h2sv1"                         ,
      "UseOlderVersion"                   , "use older version"                 ,
      "UseRandom"                         , "use random"                        ,
      "UsingFPE"                          , "using fpe"                         ,
      "VBdecayFactor"                     , "vb decay factor"                   ,
      "VBdims"                            , "vb dims"                           ,
      "VBignoreBoring"                    , "vb ignore boring"                  ,
      "VBmaxNormType"                     , "vb max norm type"                  ,
      "VBobjectBased"                     , "vb object based"                   ,
      "VBtimePeriod"                      , "vb time period"                    ,
      "VCC4maxAngle"                      , "vcc4 max angle"                    ,
      "VCC4pulseRatio"                    , "vcc4 pulse ratio"                  ,
      "VCC4serialDevice"                  , "vcc4 serial device"                ,
      "VCC4unitNo"                        , "vcc4 unit no"                      ,
      "VCEMdelay"                         , "vcem delay"                        ,
      "VCEMeyeFnames"                     , "vcem eye fnames"                   ,
      "VCEMforgetFac"                     , "vcem forget fac"                   ,
      "VCEMsaccadeOnly"                   , "vcem saccade only"                 ,
      "VCEMsigma"                         , "vcem sigma"                        ,
      "VCEMuseMax"                        , "vcem use max"                      ,
      "VCXSurpriseExp"                    , "vcx surprise exp"                  ,
      "VCXSurpriseSemisat"                , "vcx surprise semisat"              ,
      "VCXSurpriseThresh"                 , "vcx surprise thresh"               ,
      "VCXloadOutFrom"                    , "vcx load out from"                 ,
      "VCXnormSurp"                       , "vcx norm surp"                     ,
      "VCXsaveOutTo"                      , "vcx save out to"                   ,
      "VisualCortexOutputFactor"          , "visual cortex output factor"       ,
      "VisualCortexSaveOutput"            , "visual cortex save output"         ,
      "VisualCortexSurpriseType"          , "visual cortex surprise type"       ,
      "VisualCortexType"                  , "visual cortex type"                ,
      "WTAsaveResults"                    , "wta save results"                  ,
      "WTAuseBlinkSuppress"               , "wta use blink suppress"            ,
      "WTAuseSacSuppress"                 , "wta use sac suppress"              ,
      "WaitForUser"                       , "wait for user"                     ,
      "WinnerTakeAllGreedyThreshFac"      , "winner take all greedy thresh fac" ,
      "WinnerTakeAllType"                 , "winner take all type"              ,
      "XFullImplementation"               , "x full implementation"             ,
      "XptSavePrefix"                     , "xpt save prefix"                   ,
      "ZeroNumberFrames"                  , "zero number frames"                ,
      "hflip"                             , "hflip"                             ,
      "icamatrix"                         , "icamatrix"                         ,
      "localconfig"                       , "localconfig"                       ,
      "name"                              , "name"                              ,
      0
    };

  for (size_t i = 0; words[i] != 0; i += 2)
    {
      REQUIRE_EQ(camelCaseToSpaces(words[i], &acronyms),
                 std::string(words[i+1]));
    }
}

namespace
{
  // try to convert the string to type T, return true if the
  // conversion FAILED
  template <class T>
  bool conversionFails(const std::string& str)
  {
    try {
      (void) fromStr<T>(str);
    }
    catch (conversion_error& e) {
      return true;
    }
    return false;
  }
}

static void stringconversions_xx_int_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(toStr(int(-12345)), "-12345");

  REQUIRE_EQ(fromStr<int>("  456"), 456); // leading space ok
  REQUIRE_EQ(fromStr<int>("789  "), 789); // trailing space ok

  REQUIRE(conversionFails<int>("1.")); // trailing junk NOT ok
  REQUIRE(conversionFails<int>("1 a")); // trailing junk NOT ok
  REQUIRE(conversionFails<int>("1-")); // trailing junk NOT ok
  REQUIRE(conversionFails<int>("-"));
  REQUIRE(conversionFails<int>("-abc"));
  REQUIRE(conversionFails<int>(".2"));
}

static void stringconversions_xx_unsigned_char_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(toStr((unsigned char)(255)), "255");

  REQUIRE(conversionFails<unsigned char>("256"));
  REQUIRE(conversionFails<unsigned char>("-1"));
  REQUIRE(conversionFails<unsigned char>(".2"));
}

static void stringconversions_xx_unsigned_int_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(toStr((unsigned int)(67890)), "67890");

  // somehow this passes on mandriva 2010 / gcc 4.4.1
  //REQUIRE(conversionFails<unsigned int>("-12345"));
  REQUIRE(conversionFails<unsigned int>(".2"));
}

static void sortutil_xx_sortrank_xx_1(TestSuite& suite)
{
  std::vector<float> v;
  v.push_back(5.0F);
  v.push_back(6.0F);
  v.push_back(4.0F);
  v.push_back(7.0F);

  std::vector<size_t> ranks;

  util::sortrank(v, ranks);

  REQUIRE_EQ(int(ranks.size()), 4);
  REQUIRE_EQ(int(ranks[0]), 2);
  REQUIRE_EQ(int(ranks[1]), 0);
  REQUIRE_EQ(int(ranks[2]), 1);
  REQUIRE_EQ(int(ranks[3]), 3);
}

static void sortutil_xx_sortindex_xx_1(TestSuite& suite)
{
  std::vector<float> v,b;
  v.push_back(5.0F);
  v.push_back(6.0F);
  v.push_back(4.0F);
  v.push_back(7.0F);

  std::vector<size_t> ix;

  util::sortindex(v,b,ix);

  REQUIRE_EQ(int(ix.size()), 4);
  REQUIRE_EQ(int(ix[0]), 1);
  REQUIRE_EQ(int(ix[1]), 2);
  REQUIRE_EQ(int(ix[2]), 0);
  REQUIRE_EQ(int(ix[3]), 3);
}
///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(sformat_xx_format_xx_1);
  suite.ADD_TEST(sformat_xx_format_xx_2);
  suite.ADD_TEST(stringutil_xx_split_xx_1);
  suite.ADD_TEST(stringutil_xx_split_xx_2);
  suite.ADD_TEST(stringutil_xx_split_xx_3);
  suite.ADD_TEST(stringutil_xx_split_xx_4);
  suite.ADD_TEST(stringutil_xx_split_xx_5);
  suite.ADD_TEST(stringutil_xx_split_xx_6);
  suite.ADD_TEST(stringutil_xx_join_xx_1);
  suite.ADD_TEST(stringutil_xx_join_xx_2);
  suite.ADD_TEST(stringutil_xx_join_xx_3);
  suite.ADD_TEST(stringutil_xx_join_xx_4);
  suite.ADD_TEST(stringutil_xx_tolowercase_xx_1);
  suite.ADD_TEST(stringutil_xx_touppercase_xx_1);
  suite.ADD_TEST(stringutil_xx_levenshtein_xx_1);
  suite.ADD_TEST(stringutil_xx_camelcase_xx_1);

  suite.ADD_TEST(stringconversions_xx_int_xx_1);
  suite.ADD_TEST(stringconversions_xx_unsigned_char_xx_1);
  suite.ADD_TEST(stringconversions_xx_unsigned_int_xx_1);

  suite.ADD_TEST(sortutil_xx_sortrank_xx_1);
  suite.ADD_TEST(sortutil_xx_sortindex_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // WHITEBOX_STRINGS_C_DEFINED
