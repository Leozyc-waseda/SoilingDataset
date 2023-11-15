#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <Qt/qapplication.h>
#include <QtGui/qgroupbox.h>
#include <QtGui/qgraphicsscene.h>
#include <QtGui/qgraphicsview.h>
#include <QtGui/qtableview.h>
#include <QtGui/qslider.h>
#include <QtGui/qboxlayout.h>
#include <QtGui/qmainwindow.h>
#include <QtGui/qpushbutton.h>
#include <QtGui/qmenubar.h>
#include <QtGui/qmenu.h>
#include <QtGui/qfiledialog.h>
#include <QtGui/qmessagebox.h>
#include <QtGui/qtoolbar.h>
#include <QtCore/qtimeline.h>


#include "Component/ModelManager.H"
#include "Component/ModelComponent.H"
#include "Media/FrameSeries.H"
#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include "QtUtil/ImageConvert4.H"
#include "Qt4/ImageGraphicsItem.qt.H"
#include "NeovisionII/NeoAnnotate/CachedFrameLoader.H"
#include "NeovisionII/NeoAnnotate/MainDisplay.qt.H"
#include "NeovisionII/NeoAnnotate/AnnotationObjectManager.qt.H"
#include "NeovisionII/NeoAnnotate/DBManager.qt.H"
#include "NeovisionII/NeoAnnotate/PreferencesDialog.qt.H"

class MainWindow : public QMainWindow
{
  Q_OBJECT

  public:
    //! Create a new MainWindow
    MainWindow();

    //! Constructs the actual window
    /*! buildWindow() does all the work of actually instantiating all of the Qt
     * member objects and setting up the layouts, etc. Any changes to the
     * window building should go in here.*/
    void buildWindow();

    bool eventFilter(QObject * watched, QEvent * event);

    void setDBStatusLabel(QString text);
    void setAnnotatorLabel(QString text);
    FrameRange getFrameRange() { return itsCachedFrameLoader->getFrameRange(); }
    MainDisplay* getMainDisplay() { return itsMainDisplay; }
    QTimeLine* getTimeline() { return itsTimeline; }

  public slots:
    //! Update a frame on a new timeline event
    void updateFrame(int frameNum);

    //! Create a new Annotation Object, and pass it to our Main Display and Object Manager
    void addObject();

    void playPushed();

    void changeTime(int frameNum);

    void animationViewPopup(const QPoint & pos);

    void animationFrameSelected(const QModelIndex &);

    //! Open a dialog to export the annotation data to xml
    //void saveAnnotation();

    //! Open a dialog to load the annotation data from xml
    //void loadAnnotation();

    //! Open a new video file
    void openVideo();

    void openVideo(QString fileName);

    //! Mark whether the timeline was playing or not, and pause it
    void sliderPressed();

    //! Check whether the timeline was playing when the slider was pressed, and resume it if so
    void sliderReleased();

    void saveAnnotationToDB();

    void openAnnotationFromDB();

    void createDBEntry();

    QString getFileName() { return itsFileName; }

    AnnotationObjectManager* getObjectManager() { return itsObjectManager; }

    void openPrefsDialog();

  signals:
    //! Pause the current playback
    void pausePlayback(bool);

    void frameIndexChanged(int);

  private:
    /*
    //! Exports the current annotation to XML
    void saveAnnotationToXML(QString directoryName, std::map<int, std::map<int, AnnotationObjectFrame > > animation);

    //! Loads an annotation state from XML
    void loadAnnotationFromXML(QString Directory);
    */

    //! Create the File/Edit/etc.. menu bar
    void createMenuBar();

    //! Create the timeline which will drive the movie and animations
    void createTimeline();

    //! Create the top toolbar
    /*! Initializes all of the tool buttons such as zoom, add vertex, etc. that
     *  are applicable to the main display*/
    void createToolbar();

    //! Creates the transport box and returns a pointer to the box widget containing it
    QGroupBox* createTransport();

    //! Create the object list view and return an overarching widget containing the associated bits
    QWidget* createObjectList();

    QTableView* createAnimationControls();

    //! An animation timeline to control the current frame
    QTimeLine *itsTimeline;

    //! A progress bar showing/controlling the current frame
    QSlider *itsProgressBar;

    //!A label to show the current frame
    QLabel *itsFrameLabel;

    QPushButton *itsPlayButton;

    QTableView *itsAnimationView;

    //! The main image and object display
    MainDisplay       *itsMainDisplay;

    //! The repository for all of our annotation objects
    AnnotationObjectManager *itsObjectManager;

    DBManager itsDBManager;

    //! A wrapper around InputFrameSeries
    nub::ref<CachedFrameLoader> itsCachedFrameLoader;

    //! The movie playback speed in frames / second
    float itsFramerate;

    //! The current frame, set in updateFrame()
    int itsCurrentFrame;

    //! The total number of frames
    int itsNumFrames;

    bool timelineWasRunning;

    QString itsFileName;

    QLabel itsDBStatusLabel;
    QLabel itsAnnotatorLabel;

    PreferencesDialog itsPrefsDialog;
    QSettings *itsSettings;
};

#endif //MAINWINDOW_H

