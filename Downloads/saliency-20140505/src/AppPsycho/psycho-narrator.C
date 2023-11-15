/*!@file AppPsycho/psycho-narrator.C Psychophysics interactive display of
      still images or movies with speech recording */

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
// Primary maintainer for this file: Jinyong Lee <jinyongl@usc.edu>
// $HeadURL:
// $Id: psycho-narrator.C
//

#include "Component/ModelManager.H"
#include "Component/ComponentOpts.H"
#include "Component/ModelOptionDef.H"
#include "Component/EventLog.H"

#include "Image/Image.H"
#include "Raster/Raster.H"
#include "Video/VideoFrame.H"
#include "Util/Types.H"
#include "Util/MathFunctions.H"
#include "Util/FileUtil.H"
#include "Util/sformat.H"
#include "rutz/time.h"

#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "GUI/GUIOpts.H"

#include "Devices/AudioGrabber.H"
#include "Devices/AudioMixer.H"
#include "Devices/DeviceOpts.H"
#include "Audio/AudioWavFile.H"

#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"
#include "Neuro/NeuroOpts.H"

#include <vector>
#include <pthread.h>
#include <signal.h>


// ######################################################################
// Definitions & Constants

#define CACHELEN						150

// supported visual stimulus file extensions (adopted from "Media/MediaOpts.C")
// needed to be constantly updated unless there's a way to check automatically
//
const char *imageExtensions[] = { ".pnm", ".pgm", ".ppm", ".pbm", ".pfm", ".png", ".jpeg", ".jpg", ".dpx", NULL };
const char *movieExtensions[] = { ".avi", ".mpg", ".mpeg", ".m4v", ".m2v", ".mov", ".flv", ".dv", NULL };

// experiment procedure type
#define PROC_NORMAL						0
#define PROC_EXP1						1		// online-offline switch
#define PROC_EXP2						2		// scene pairs & online-offline switch
#define PROC_EXP3						3		// change blindness paradigm
#define PROC_EXP4						4		// scene pairs & low-high threshold switch
#define PROC_INVALID					-1

// input stimulus type
#define STIM_IMAGE						0
#define STIM_MOVIE						1
#define STIM_UNKNOWN					-1


// recording styles
#define REC_NONE						0
#define REC_DURING						(1 << 0)
#define REC_AFTER						(1 << 1)
#define REC_ALL							(REC_DURING | REC_AFTER)


// session structures
struct SESSION_EXP12
{
	uint	index;
	uint	staticPeriod;
	uint	blankPeriod;
	
	const char* flag;
	const char* message;
};

struct SESSION_EXP3
{
	uint	index;
	uint	style;

	const char* flag;
};

struct SESSION_EXP4
{
	uint	index;
	uint	task;
	bool	practice;

	const char* flag;
	const char* reminder;
};

//
// psycho-narrator options
//

static const ModelOptionDef OPT_ProcedureType =
	{ MODOPT_ARG_STRING, "ProcedureType", &MOC_DISPLAY, OPTEXP_CORE,
		"Use experiment specific procedure types that would override relevant parameter settings.",
	    "proc-type", '\0', "<Normal|Exp1|Exp2|Exp3|Exp4>", "Normal" };

static const ModelOptionDef OPT_EyeTrackerRecalib =
	{ MODOPT_ARG(uint), "EyeTrackerRecalibration", &MOC_EYETRACK, OPTEXP_CORE,
		"Recalibration frequency of EyeTracker. Set 0 if you don't want recalibration at all."
		"If you set to 1, then recalibration will be done after every single image session.",
		"et-recalib", '\0', "<int>", "0" };

static const ModelOptionDef OPT_ShuffleOrder =
	{ MODOPT_FLAG, "ShuffleOrder", &MOC_DISPLAY, OPTEXP_CORE,
		"Whether shuffle the order of input stimuli or not.",
		"shuffle", '\0', "<bool>", "false" };

static const ModelOptionDef OPT_MouseInput =
	{ MODOPT_FLAG, "MouseInput", &MOC_DISPLAY, OPTEXP_CORE,
		"Make mouse input available and use it instead of key presses.",
		"mouse-input", '\0', "<bool>", "false" };

static const ModelOptionDef OPT_BlankPeriod =
	{ MODOPT_ARG(uint), "BlankPeriod", &MOC_DISPLAY, OPTEXP_CORE,
		"The period (in sec) of the blank sessions between the visual stimulus presentations. "
		"If set to 0, no blank session will be presented. If set bigger than 999,"
		"blank session continues until the user presses mouse button or a key.",
		"blank-period", '\0', "<int>", "5" };

static const ModelOptionDef OPT_StaticPeriod =
	{ MODOPT_ARG(uint), "StaticImagePeriod", &MOC_DISPLAY, OPTEXP_CORE,
		"The period (in sec) of static images (if the currently showing image is a raster file) "
		"during which they are presented on the screen. For video clips, this option is ignored.",
		"static-period", '\0', "<int>", "5" };

static const ModelOptionDef OPT_EyeTrackerRec =
	{ MODOPT_ARG_STRING, "EyeTrackerRecordStyle", &MOC_EYETRACK, OPTEXP_SAVE,
		"During when eye-tracker data should be grabbed and recorded. "
		"'During' means the recording is done only during the stimulus presentation whereas "
		"'All' is done even after the presentation - i.e. including the blank presentation.",
		"et-rec", '\0', "<During|All>", "During" };

static const ModelOptionDef OPT_AudRec =
	{ MODOPT_ARG_STRING, "AudioRecordStyle", &MOC_DISPLAY, OPTEXP_SAVE,
		"During when audio(speech) should be grabbed and recorded. "
		"'During' means the recording is done only during the stimulus presentation whereas "
		"'After' is done only after the presentation - i.e. during the blank sessions.",
		"aud-rec", '\0', "<None|During|After|All>", "All" };
/*
static const ModelOptionDef OPT_ExpName =
	{ MODOPT_ARG_STRING, "ExperimentName", &MOC_DISPLAY, OPTEXP_SAVE,
		"The name of experiment. "
		"Event log and recorded speech will be saved as the name specified by this option.",
		"exp-name", '\0', "<name>", "exp_narrator" };
*/


// thread mutex variable key for audio recording
static pthread_mutex_t audMutexKey = PTHREAD_MUTEX_INITIALIZER;
volatile bool audExit = false;					// audio thread exit flag
volatile bool audRec = false;					// grabbed audio recording flag

std::vector< AudioBuffer<byte> > rec;			// grabbed audio buffer


// ######################################################################
// Submodules

// case insensitive conversion of record style
int readRecordStyle( const std::string& recStr )
{
	if( strcasecmp( recStr.c_str(), "During" ) == 0 )
		return REC_DURING;
	
	if( strcasecmp( recStr.c_str(), "After" ) == 0 )
		return REC_AFTER;
	
	if( strcasecmp( recStr.c_str(), "All" ) == 0 )
		return REC_ALL;
	
	return REC_NONE;	// no-recording by default
}

// case insensitive conversion of procedure type
int readProcType( const std::string& procType )
{
	if( strcasecmp( procType.c_str(), "Normal" ) == 0 )
		return PROC_NORMAL;
	
	if( strcasecmp( procType.c_str(), "Exp1" ) == 0 )
		return PROC_EXP1;
	
	if( strcasecmp( procType.c_str(), "Exp2" ) == 0 )
		return PROC_EXP2;

	if( strcasecmp( procType.c_str(), "Exp3" ) == 0 )
		return PROC_EXP3;

	if( strcasecmp( procType.c_str(), "Exp4" ) == 0 )
		return PROC_EXP4;
	
	return PROC_INVALID;	// invalid procedure type
}

// get stimulus file type according to the extension
int getStimulusType( const std::string& fname )
{
	// check if it is a raster file
	for( int i = 0; imageExtensions[i]; i ++ )
		if( hasExtension( fname, imageExtensions[i] ))
			return STIM_IMAGE;
	
	// check if it is a movie file
	for( int i = 0; movieExtensions[i]; i ++ )
		if( hasExtension( fname, movieExtensions[i] ))
			return STIM_MOVIE;
	
	return STIM_UNKNOWN;	// unknown file type
}

// extract stimulus name field only
void getStimulusName( const std::string &stimPath, std::string &stimName )
{
		std::string name, path;
		splitPath( stimPath, path, name );
		std::string::size_type dot = name.rfind( '.' );
		if( dot != name.npos ) name.erase( dot );

		stimName = name;
}

// pause until a key/mouse button pressed
void pause( bool mouse, nub::soft_ref<PsychoDisplay>& d )
{
	if( mouse )
		d->waitForMouseClick();
	else
		d->waitForKey();
}

// sleep while checking key press
void snooze( uint sec, nub::soft_ref<PsychoDisplay>& d )
{
	if( sec < 1 ) return;
	
	rutz::time start = rutz::time::wall_clock_now();
	rutz::time stop;
	
	do {
		d->checkForKey();
		usleep( 50000 );	// wait for 0.05 sec
		stop = rutz::time::wall_clock_now();
	
	} while( (uint)(stop-start).sec() < sec );
}

// start/stop eye-tracking
void trackEyes( bool trk, nub::soft_ref<EyeTracker>& et, nub::soft_ref<PsychoDisplay>& d )
{
	if( trk )
	{
		if( !et->isTracking() )
		{
			// start the eye tracker
			et->track( true );
			
			// blink the fixation point at the center
			d->displayFixationBlink();
		}
	} else
	{
		if( et->isTracking() )
		{
			// stop the eye tracker
			usleep( 50000 );	// wait for 0.05 sec
			et->track( false );
		}
	}
}

// start/stop audio recording
void recordAudio( bool rec, nub::soft_ref<PsychoDisplay>& d )
{
	if( rec )
	{
		if( !audRec ) d->pushEvent( "---- Audio Recording Start ----" );
		audRec = true;
	} else
	{
		if( audRec ) d->pushEvent( "---- Audio Recording Stop ----" );
		audRec = false;
	}
}

// calibrate ISCAN
void calibrateISCAN( bool mouse, nub::soft_ref<PsychoDisplay>& d )
{
	/*
	et->calibrate(d);
	d->clearScreen();
	*/

	d->clearScreen();
	d->displayText( "ISCAN calibration" );
	pause( mouse, d );

	// display ISCAN calibration grid
	d->clearScreen();
	d->displayISCANcalib();
	pause( mouse, d );

	// run 15-point calibration
	d->clearScreen();
	
	if( mouse )
	{
		d->displayText( "click LEFT button to calibrate or RIGHT button to skip" );
		int ret = d->waitForMouseClick();
		if( ret == 1 ) d->displayEyeTrackerCalibration( 3, 5, 1, true );
	} else
	{
		d->displayText( "press SPACE key to calibrate or other key to skip" );
		int ret = d->waitForKey();
		if( ret == ' ' ) d->displayEyeTrackerCalibration( 3, 5, 1 );
	}
	
	d->clearScreen();
}

// buffering movie frames
static bool cacheFrame( nub::soft_ref<InputMPEGStream>& mp, std::deque<VideoFrame>& cache )
{
	const VideoFrame frame = mp->readVideoFrame();
	if( !frame.initialized() )
		return false;	// end of stream
	
	cache.push_front( frame );
	return true;
}

// save recorded speech into a wave file
void saveAudioRecord( const std::string &wavname, nub::soft_ref<PsychoDisplay>& d )
{
	pthread_mutex_lock( &audMutexKey );
	
	if( rec.size() > 0 )	// if there is some record to write
	{
		LINFO("Saving '%s'...", wavname.c_str());

		// write audio file
		d->pushEventBegin( std::string("writeAudioFile: '") + wavname + "'" );
		writeAudioWavFile( wavname, rec );
		d->pushEventEnd( "writeAudioFile" );
		
		rec.clear();	// reset recorded audio
	}

	pthread_mutex_unlock( &audMutexKey );
}

// audio grabbing thread function
static void *grabAudio( void *arg )
{
	LINFO("Initiating the audio-grabbing thread...");

	AudioGrabber *pAgb = (AudioGrabber*)arg;
	
	// mask all allowed signals
	sigset_t mask;	
	sigfillset( &mask );
  	if( pthread_sigmask( SIG_BLOCK, &mask, NULL ) != 0 )
  		LINFO("Failed to mask signals for the audio-grabbing thread!");

	// grab audio data endlessly
	while( !audExit )
	{
		// grab audio data from input buffer
		AudioBuffer<byte> data;
		pAgb->grab( data );

		// record grabbed audio
		if( audRec )
		{
			pthread_mutex_lock( &audMutexKey );
			rec.push_back( data );
			pthread_mutex_unlock( &audMutexKey );
		}
	}
	
	LINFO("Quitting the audio-grabbing thread...");
	
	return NULL;
}


// ######################################################################
// psycho-narrator main function
//

static int submain( const int argc, char** argv )
{
	MYLOGVERB = LOG_INFO;  // suppress debug messages

	// create ModelManager
	ModelManager manager( "Psycho Narrator" );

	// hook up newly created psycho-narrator options
	OModelParam<std::string> procTypeStr( &OPT_ProcedureType, &manager );
	OModelParam<uint> etRecalib( &OPT_EyeTrackerRecalib, &manager );
	OModelParam<bool> shuffle( &OPT_ShuffleOrder, &manager );
	OModelParam<bool> mouseInput( &OPT_MouseInput, &manager );
	OModelParam<uint> blankPeriod( &OPT_BlankPeriod, &manager );
	OModelParam<uint> staticPeriod( &OPT_StaticPeriod, &manager );
	OModelParam<std::string> audRecStr( &OPT_AudRec, &manager );
	OModelParam<std::string> etRecStr( &OPT_EyeTrackerRec, &manager );
//	OModelParam<std::string> expName( &OPT_ExpName, &manager );

	//
	// create various ModelComponents
	//

	// display
	nub::soft_ref<PsychoDisplay> d( new PsychoDisplay(manager) );
	manager.addSubComponent( d );

	// event log
	nub::soft_ref<EventLog> el( new EventLog(manager) );
	manager.addSubComponent( el );

	// eye-tracker configurator
	nub::soft_ref<EyeTrackerConfigurator> etc( new EyeTrackerConfigurator(manager) );
	manager.addSubComponent( etc );

	// MPEG stream
	nub::soft_ref<InputMPEGStream> mp( new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream") );
	manager.addSubComponent( mp );

	// audio mixer
	nub::soft_ref<AudioMixer> amx( new AudioMixer(manager) );
	manager.addSubComponent( amx );

	// audio grabber
	nub::soft_ref<AudioGrabber> agb( new AudioGrabber(manager) );
	manager.addSubComponent( agb );

	// audio buffer
	std::vector< AudioBuffer<byte> > rec;

	// set option defaults
	manager.setOptionValString( &OPT_SDLdisplayDims, "1920x1080" );
	manager.setOptionValString( &OPT_EventLogFileName, "narrator.psy" );
	manager.setOptionValString( &OPT_EyeTrackerType, "ISCAN" );
	manager.setOptionValString( &OPT_InputMPEGStreamPreload, "true" );
	manager.setOptionValString( &OPT_AudioMixerLineIn, "false" );
	manager.setOptionValString( &OPT_AudioMixerCdIn, "false" );
	manager.setOptionValString( &OPT_AudioMixerMicIn, "true" );
	manager.setOptionValString( &OPT_AudioGrabberBits, "8" );
	manager.setOptionValString( &OPT_AudioGrabberFreq, "11025" );
	manager.setOptionValString( &OPT_AudioGrabberBufSamples, "256" );
	manager.setOptionValString( &OPT_AudioGrabberChans, "1" );

	// parse command-line
	if( manager.parseCommandLine( argc, argv, "<stimulus 1> ... <stimulus N>", 1, -1 ) == false )
		return 1;

//	manager.setOptionValString( &OPT_EventLogFileName, expName.getVal() + ".psy" );
		
	int etRecStyle = readRecordStyle( etRecStr.getVal() );
	int audRecStyle = readRecordStyle( audRecStr.getVal() );
	
	int procType = readProcType( procTypeStr.getVal() );
	if( procType == PROC_INVALID )
		LFATAL("Invalid procedure type '%s'", procTypeStr.getVal().c_str());		

	// hook up ModelComponents
	nub::soft_ref<EyeTracker> et = etc->getET();
	d->setEyeTracker( et );
	d->setEventLog( el );
	et->setEventLog( el );

	if( !(audRecStyle & REC_ALL) )   // if no audio recording, remove audio components
	{
		manager.removeSubComponent( *amx, true );
		manager.removeSubComponent( *agb, true );
	}

	// initiate ModelComponent instances
	manager.start();

	// create audio-grabbing thread
	// NOTE: audio thread should start here right after the grabber started
	pthread_t audGrbId;
	if( audRecStyle & REC_ALL )		// if audio recording, initiate the audio thread
	{
		pthread_create( &audGrbId, NULL, &grabAudio, (void*)agb.get() );
	}
	
	//
	// begin experiment
	//

	// eye tracker calibration
	calibrateISCAN( mouseInput.getVal(), d );

	//
	// start experiment procedure
	//
	switch( procType )
	{
		
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	case PROC_NORMAL:
		LINFO("Normal experiment procedure...");

		{
			uint recalibCount = 0;
			uint stimNum = manager.numExtraArgs();
			uint stimIndices[stimNum];
			for( uint i = 0; i < stimNum; i ++ ) stimIndices[i] = i;
			if( shuffle.getVal() )
			{
				LINFO("Shuffling stimulus display order...");
				randShuffle( stimIndices, stimNum );
			}

			// session loop
			for( uint i = 0; i < stimNum; i ++ )
			{
				// get the type of stimulus file according to its extension
				std::string stimPath = manager.getExtraArg(stimIndices[i]);
				std::string stimName;
				getStimulusName( stimPath, stimName );
				int stimType = getStimulusType( stimPath );
				if( stimType == STIM_UNKNOWN )
					LFATAL("Unknown stimulus file extension '%s'", stimPath.c_str());

				SDL_Surface *surf = NULL;
				std::deque<VideoFrame> cache;
				bool streaming = true;
		
				d->clearScreen();
		
				// load visual stimulus
				if( stimType == STIM_IMAGE )
				{
					// load raster image
					LINFO("Loading '%s'...", stimPath.c_str());
					Image< PixRGB<byte> > image = Raster::ReadRGB( stimPath );
					surf = d->makeBlittableSurface( image, true );		
				} else
				{
					// cache initial movie frames
					LINFO("Buffering '%s'...", stimPath.c_str());
					mp->setFileName( stimPath );
					for( uint j = 0; j < CACHELEN; j ++ )
					{
						streaming = cacheFrame( mp, cache );
						if( streaming == false ) break;  // all movie frames got cached!
					}
				}
				LINFO("'%s' ready.", stimPath.c_str());
		
				// give a chance to other processes (useful on single-CPU machines)
				snooze( 1, d );
				if (system( "sync" ) == -1) LERROR("sync() failed!");
		
				// start session
				d->displayText( sformat( "Session %04d", i + 1 ) );
				pause( mouseInput.getVal(), d );
		
				d->waitNextRequestedVsync( false, true );
				if( stimType == STIM_IMAGE )
					d->pushEvent( std::string("===== Showing image: ") + stimPath + " =====" );
				else
					d->pushEvent( std::string("===== Playing movie: ") + stimPath + " =====" );
		
				// start eye tracking
				trackEyes( true, et, d );

				// show visual stimulus
				if( stimType == STIM_IMAGE )
				{
					// display image
					d->displaySurface( surf, -2 );

					// speech recording during image presentation
					recordAudio( audRecStyle & REC_DURING, d );

					// wait for a while as image is being displayed
					snooze( staticPeriod.getVal(), d );

					SDL_FreeSurface( surf );

				} else
				{
					// create an overlay
					d->createVideoOverlay( VIDFMT_YUV420P, mp->getWidth(), mp->getHeight() );
		
					// play movie
					uint frame = 0;
					rutz::time start = rutz::time::wall_clock_now();	// start time
		
					while( cache.size() )
					{
						d->checkForKey();
						
						// cache one more frame
						if( streaming )
							streaming = cacheFrame( mp, cache );
						
						// get next frame to display and put it into overlay
						VideoFrame vidframe = cache.back();
						d->displayVideoOverlay( vidframe, frame, SDLdisplay::NEXT_VSYNC );
						cache.pop_back();
						
						// speech recording during video play
						recordAudio( audRecStyle & REC_DURING, d );
						
						frame ++;
					}
					rutz::time stop = rutz::time::wall_clock_now();		// end time
					double secs = (stop - start).sec();
					LINFO("%d frames in %.02f sec (~%.02f fps)", frame, secs, frame / secs);
		
					// destroy the overlay. Somehow, mixing overlay displays and
					// normal displays does not work. With a single overlay created
					// before this loop and never destroyed, the first movie plays
					// ok but the other ones don't show up
					d->destroyYUVoverlay();
					d->clearScreen();  // sometimes 2 clearScreen() are necessary
				}
			
				d->clearScreen();
		
				// display blank
				if( blankPeriod.getVal() > 0 )
				{
					d->pushEvent( std::string("===== Presenting blank =====") );
		
					// eye tracking during blank
					trackEyes( etRecStyle & REC_AFTER, et, d );
				
					// speech recording during blank
					recordAudio( audRecStyle & REC_AFTER, d );
				
					// wait for a while as blank is being presented
					if( blankPeriod.getVal() < 1000 )
					{
						snooze( blankPeriod.getVal(), d );
					} else
					{
						pause( mouseInput.getVal(), d );
					}
				}

				// stop eye tracking
				trackEyes( false, et, d );
			
				// stop speech recording
				recordAudio( false, d );

				// save recorded speech to a wave file
// 				std::string wavname( sformat("%s-%04d.wav", expName.getVal().c_str(), i + 1) );
				saveAudioRecord( stimName + ".wav", d );
		
				// eye tracker recalibration
				if( etRecalib.getVal() > 0 )
				{
					recalibCount ++;
					if( recalibCount == etRecalib.getVal() )
					{
						recalibCount = 0;
						calibrateISCAN( mouseInput.getVal(), d );
					}
				}
			}
		}
		break;
		
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
	case PROC_EXP1:
	case PROC_EXP2:
		if( procType == PROC_EXP1 )
			LINFO("Experiment 1 procedure...");
		else
			LINFO("Experiment 2 procedure...");
			
		{
			uint sessionNum;
			uint pairNum[2];

			if( procType == PROC_EXP1 )
			{
				sessionNum = manager.numExtraArgs();
			} else
			{
				for( int i = 0; i < 2; i ++ )
				{
					pairNum[i] = (uint)atoi( manager.getExtraArg(i).c_str() );
					if( pairNum[i] < 1 )
						LFATAL("Invalid stimulus information '%s'", manager.getExtraArg(i).c_str());
				}
				sessionNum = pairNum[0] + pairNum[1];
			}
			
			SESSION_EXP12 sessions[sessionNum];

			if( procType == PROC_EXP1 )
			{
				// switching between online and offline for a pair of scenes
				for( uint i = 0; i < sessionNum; i += 2 )
				{
					uint i1, i2;
					if( randomUpToNotIncluding( 2 ) )
					{
						i1 = i;
						i2 = i + 1;
					} else
					{
						i1 = i + 1;
						i2 = i;
					}

					sessions[i1].index = i1;
					sessions[i1].staticPeriod = 15;
					sessions[i1].blankPeriod = 5;
					sessions[i1].flag = "ON";
					sessions[i1].message = "On-line Description";

					sessions[i2].index = i2;
					sessions[i2].staticPeriod = 10;
					sessions[i2].blankPeriod = 15;
					sessions[i2].flag = "OFF";
					sessions[i2].message = "Post-scene Description";
				}
			} else
			{
				// select one among a pair of scenes and switch between online and offline
				uint idx = 0;
				for( int i = 0; i < 2; i ++ )
				{
					int offset[pairNum[i]];
					int task[pairNum[i]];
					for( uint j = 0; j < pairNum[i]; j ++ )
					{
						if( j < pairNum[i] / 2 )
							offset[j] = task[j] = 0;
						else
							offset[j] = task[j] = 1;
					}
					randShuffle( offset, pairNum[i] );
					randShuffle( task, pairNum[i] );
					
					for( uint j = 0; j < pairNum[i]; j ++ )
					{						
						sessions[idx].index = 2 + idx * 2 + offset[j];
						if( task[j] == 0 )												
						{
							sessions[idx].staticPeriod = 12;
							sessions[idx].blankPeriod = 5;
							sessions[idx].flag = "ON";
							sessions[idx].message = "Describe as soon as possible";
						} else
						{
							sessions[idx].staticPeriod = 7;
							sessions[idx].blankPeriod = 10;
							sessions[idx].flag = "OFF";
							sessions[idx].message = "Describe in one sentence";
						}
						
						idx ++;
					}
				}	
			}

			// randomize session order
			randShuffle( sessions, sessionNum );

			// session loop
			for( uint i = 0; i < sessionNum; i ++ )
			{
				d->clearScreen();

				// load raster image
				std::string stimPath = manager.getExtraArg(sessions[i].index);
				std::string stimName;
				getStimulusName( stimPath, stimName );
				LINFO("Loading '%s'...", stimPath.c_str());
				Image< PixRGB<byte> > image = Raster::ReadRGB( stimPath );
				SDL_Surface *surf = d->makeBlittableSurface( image, true );
				LINFO("'%s' ready.", stimPath.c_str());

				// give a chance to other processes (useful on single-CPU machines)
				snooze( 1, d );
				if (system( "sync" ) == -1) LERROR("sync() failed!");
		
				// start session
				d->displayText( sformat( "%s", sessions[i].message ) );
				pause( mouseInput.getVal(), d );
		
				d->waitNextRequestedVsync( false, true );
				d->pushEvent( std::string("===== Showing image: ") + stimPath + " " + sessions[i].flag + " =====" );

				// start eye tracking
				trackEyes( true, et, d );

				// start speech recording
				recordAudio( true, d );

				// display image
				d->displaySurface( surf, -2 );

				// wait for a while as image is being displayed
				snooze( sessions[i].staticPeriod, d );

				SDL_FreeSurface( surf );
				d->clearScreen();

				// display blank
				d->pushEvent( std::string("===== Presenting blank =====") );

				// wait for a while as blank is being presented
				snooze( sessions[i].blankPeriod, d );

				// stop eye tracking
				trackEyes( false, et, d );

				// stop speech recording
				recordAudio( false, d );

				// save recorded speech to a wave file
				saveAudioRecord( stimName + ".wav", d );
			}
		}
		break;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
	case PROC_EXP3:
		LINFO("Experiment 3 procedure...");
		
		{
			uint sessionNum;
			uint pairNum[3];

			for( int i = 0; i < 3; i ++ )
			{
				pairNum[i] = (uint)atoi( manager.getExtraArg(i).c_str() );
				if( pairNum[i] < 1 )
					LFATAL("Invalid stimulus information '%s'", manager.getExtraArg(i).c_str());
			}
			sessionNum = pairNum[0] + pairNum[1] + pairNum[2];

			SESSION_EXP3 sessions[sessionNum];

			uint idx = 0;
			for( int i = 0; i < 3; i ++ )
			{
				int offset[pairNum[i]];
				for( uint j = 0; j < pairNum[i]; j ++ )
				{
					if( j < pairNum[i] / 2 )
						offset[j] = 0;		// normal scene
					else
						offset[j] = 1;		// faded scene
				}
				randShuffle( offset, pairNum[i] );

				for( uint j = 0; j < pairNum[i]; j ++ )
				{
					sessions[idx].style = i;
					switch( sessions[idx].style )
					{
					case 0:		// practice
						sessions[idx].index = 3 + j;
						sessions[idx].flag = "PRACTICE";
						break;

					case 1:		// standard
						sessions[idx].index = 3 + pairNum[0] + j * 2;
						if( offset[j] == 1 ) sessions[idx].index ++;
						sessions[idx].flag = "STD";
						break;

					case 2:		// change blindness
						sessions[idx].index = 3 + pairNum[0] + pairNum[1] * 2 + j * 4;
						if( offset[j] == 1 ) sessions[idx].index += 2;
						sessions[idx].flag = "CHG";
						break;
					}

					idx ++;
				}
			}

			// randomize session order (practice sessions excluded)
			randShuffle( &sessions[pairNum[0]], sessionNum - pairNum[0] );

			//
			// session loop
			//
			uint sessCnt = 1;
			uint pracCnt = 1;
			for( uint i = 0; i < sessionNum; i ++ )
			{
				d->clearScreen();

				// load raster image(s)
				std::string stimPath = manager.getExtraArg(sessions[i].index);
				std::string stimName;
				getStimulusName( stimPath, stimName );
				LINFO("Loading '%s'...", stimPath.c_str());
				Image< PixRGB<byte> > image = Raster::ReadRGB( stimPath );
				SDL_Surface *surf = d->makeBlittableSurface( image, true );
				SDL_Surface *surf_ch = NULL;
				if( sessions[i].style == 2 )
				{
					Image< PixRGB<byte> > image_ch = Raster::ReadRGB( manager.getExtraArg(sessions[i].index + 1) );
					surf_ch = d->makeBlittableSurface( image_ch, true );
				}
				LINFO("'%s' ready.", stimPath.c_str());

				// give a chance to other processes (useful on single-CPU machines)
				snooze( 1, d );
				if (system( "sync" ) == -1) LERROR("sync() failed!");

				// start session
				if( sessions[i].style == 0 )
					d->displayText( sformat( "Pactice %04d", pracCnt ) );
				else
					d->displayText( sformat( "Session %04d", sessCnt ) );
				pause( mouseInput.getVal(), d );

				d->displayText( "Describe the event(s) of scene as quickly as possible." );
				pause( mouseInput.getVal(), d );

				d->waitNextRequestedVsync( false, true );
				d->pushEvent( std::string("===== Showing image: ") + stimPath + " " + sessions[i].flag + " =====" );

				if( sessions[i].style != 0 )
				{
					// start eye tracking
					trackEyes( true, et, d );
	
					// start speech recording
					recordAudio( true, d );
				} else
				{
					// blink the fixation point at the center
					d->displayFixationBlink();
				}

				for( int j = 0; j < 10; j ++ )
				{
					// display image
					if( sessions[i].style == 2 && j >= 3 && j <= 6 )	// show changed scene from 4.5 sec to 10.5 sec (for 6 sec)
						d->displaySurface( surf_ch );		// apply change
					else
						d->displaySurface( surf, j == 0 ? -2 : -1 );
					usleep( 1000000 );	// wait for 1 sec
					
					// display field blank
					d->SDLdisplay::clearScreen( PixRGB<byte>(0, 0, 0), true );
					usleep( 500000 );	// wait for 0.5 sec

					d->checkForKey();
				}

				SDL_FreeSurface( surf );
				if( surf_ch ) SDL_FreeSurface( surf_ch );
				d->SDLdisplay::clearScreen( PixRGB<byte>(0, 0, 0), true );

				// display blank
				d->pushEvent( std::string("===== Presenting blank =====") );

				// wait for a while as blank is being presented
				snooze( 5, d );

				if( sessions[i].style != 0 )
				{
					// stop eye tracking
					trackEyes( false, et, d );

					// stop speech recording
					recordAudio( false, d );

					// save recorded speech to a wave file
					saveAudioRecord( stimName + ".wav", d );
				}

				d->displayText( "Describe what you have seen with as much detail as possible." );
				pause( mouseInput.getVal(), d );
				d->clearScreen();

				if( sessions[i].style != 0 )
				{
					// start speech recording
					recordAudio( true, d );
				}
				snooze( 3, d );		// 3 sec of grace time for press mistakes
				pause( mouseInput.getVal(), d );

				if( sessions[i].style != 0 )
				{
					// stop speech recording
					recordAudio( false, d );

					// save recorded speech to a wave file
					saveAudioRecord( stimName + "_post.wav", d );
				}
				
				if( sessions[i].style != 0 )
					sessCnt ++;
				else
					pracCnt ++;
			}
		}
		break;
		
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
	case PROC_EXP4:
		LINFO("Experiment 4 procedure...");
		
		{
			uint sessionNum;
			uint qckSessionNum, frmSessionNum;
			uint qckPracNum, frmPracNum;
			uint stimNum[3];

			for( int i = 0; i < 3; i ++ )
			{
				stimNum[i] = (uint)atoi( manager.getExtraArg(i).c_str() );
				if( stimNum[i] < 1 )
					LFATAL("Invalid stimulus information '%s'", manager.getExtraArg(i).c_str());
			}
			sessionNum = stimNum[0] + stimNum[1] + stimNum[2];			
			qckSessionNum = (stimNum[0] / 2) + (stimNum[1] / 2) + (stimNum[2] / 2);
			frmSessionNum = sessionNum - qckSessionNum;
			qckPracNum = stimNum[0] / 2;
			frmPracNum = stimNum[0] - qckPracNum;			

			SESSION_EXP4 sessions[sessionNum];

			uint qckIdx = 0;
			uint frmIdx = qckSessionNum;
			for( int i = 0; i < 3; i ++ )
			{
				int offset = 3;
				for( int j = 0; j < i; j ++ )
					offset += stimNum[j];

				int task[stimNum[i]];
				for( uint j = 0; j < stimNum[i]; j ++ )
					task[j] = (j < stimNum[i] / 2) ? 0 : 1;
				randShuffle( task, stimNum[i] );

				for( uint j = 0; j < stimNum[i]; j ++ )
				{
					if( task[j] == 0 )
					{
						// as quickly as possible
						sessions[qckIdx].index = j + offset;
						sessions[qckIdx].practice = (i == 0);
						sessions[qckIdx].task = 0;
						sessions[qckIdx].flag = "QCK";
						sessions[qckIdx].reminder = "Describe the event(s) of the scene AS QUICKLY AS POSSIBLE while watching.";
						
						qckIdx ++;
					} else
					{
						// complete and well-formed
						sessions[frmIdx].index = j + offset;
						sessions[frmIdx].practice = (i == 0);
						sessions[frmIdx].task = 1;
						sessions[frmIdx].flag = "FRM";
						sessions[frmIdx].reminder = "Describe the event(s) of the scene in COMPLETE AND WELL-FORMED SENTENCES while watching.";
						
						frmIdx ++;
					}
				}
			}

			// randomize session order (practice sessions excluded)
			randShuffle( &sessions[qckPracNum], qckSessionNum - qckPracNum );
			randShuffle( &sessions[qckSessionNum + frmPracNum], frmSessionNum - frmPracNum );

			//
			// session loop
			//
			uint curTask = 100;
			for( uint i = 0; i < sessionNum; i ++ )
			{
				d->clearScreen();

				// load raster image(s)
				std::string stimPath = manager.getExtraArg(sessions[i].index);
				std::string stimName;
				getStimulusName( stimPath, stimName );
				LINFO("Loading '%s'...", stimPath.c_str());
				Image< PixRGB<byte> > image = Raster::ReadRGB( stimPath );
				SDL_Surface *surf = d->makeBlittableSurface( image, true );
				LINFO("'%s' ready.", stimPath.c_str());

				// give a chance to other processes (useful on single-CPU machines)
				snooze( 1, d );
				if (system( "sync" ) == -1) LERROR("sync() failed!");
				
				// task description
				if( curTask != sessions[i].task )
				{
					d->displayText( sformat( "Task Description") );
					pause( mouseInput.getVal(), d );
					switch( sessions[i].task )
					{
					case 0:
						d->displayText( "The task is to describe the event(s) of the scene AS QUICKLY AS POSSIBLE." );
						pause( mouseInput.getVal(), d );
						d->displayText( "Note that in this task, you don't have to worry about" );
						pause( mouseInput.getVal(), d );
						d->displayText( "the well-formedness or grammatical correctness of the sentence." );
						pause( mouseInput.getVal(), d );
						break;
						
					case 1:
						d->displayText( "The task is to describe the event(s) of the scene in COMPLETE AND WELL-FORMED SENTENCES." );
						pause( mouseInput.getVal(), d );
						d->displayText( "Note that in this task, you don't have to speak quickly." );
						pause( mouseInput.getVal(), d );
						d->displayText( "Focus on the sentence structure while taking as much time as you want." );
						pause( mouseInput.getVal(), d );
						break;
					}
					curTask = sessions[i].task;
				}

				// start session
				if( sessions[i].practice )
					d->displayText( sformat( "Session %04d (Practice)", i + 1 ) );
				else
					d->displayText( sformat( "Session %04d", i + 1 ) );
				pause( mouseInput.getVal(), d );

				// task reminder
				d->displayText( sessions[i].reminder );
				pause( mouseInput.getVal(), d );

				d->waitNextRequestedVsync( false, true );				
				if( sessions[i].practice == false )
				{
					d->pushEvent( std::string("===== Showing image: ") + stimPath + " " + sessions[i].flag + " =====" );

					// start eye tracking
					trackEyes( true, et, d );
	
					// start speech recording
					recordAudio( true, d );
				} else
				{
					d->pushEvent( std::string("===== Showing image: ") + stimPath + " PRACTICE =====" );

					// blink the fixation point at the center
					d->displayFixationBlink();
				}

				// display image
				d->displaySurface( surf, -2 );

				// wait until description speech is done
				snooze( 3, d );		// 3 sec of grace time for press mistakes
				pause( mouseInput.getVal(), d );

				SDL_FreeSurface( surf );
				d->clearScreen();

				if( sessions[i].practice == false )
				{
					// stop eye tracking
					trackEyes( false, et, d );

					// stop speech recording
					recordAudio( false, d );

					// save recorded speech to a wave file
					saveAudioRecord( stimName + ".wav", d );
				}

				// display blank
				d->pushEvent( std::string("===== Presenting blank =====") );

				// wait for a while as blank is being presented
				snooze( 2, d );
			}
		}
		break;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}

	//
	// finish experiment
	//
	d->clearScreen();
	d->displayText( "Experiment complete." );
	pause( mouseInput.getVal(), d );

	// kill audio grabbing thread
	audExit = true;
	if( audRecStyle & REC_ALL )
		pthread_join( audGrbId, NULL );

	// stop all ModelComponents
	manager.stop();

	return 0;
}



// ######################################################################
// psycho-narrator dummy main function
//

extern "C" int main(const int argc, char** argv)
{
	// simple wrapper around submain() to catch exceptions (because we
	// want to allow PsychoDisplay to shut down cleanly; otherwise if we
	// abort while SDL is in fullscreen mode, the X server won't return
	// to its original resolution)

	try
	{
		return submain( argc, argv );
	}
	catch (...)
	{
		REPORT_CURRENT_EXCEPTION;
	}

	return 1;
}

