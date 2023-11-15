
#ifndef MAINWINDOW_C
#define MAINWINDOW_C

#include "NeovisionII/NeoAnnotate/MainWindow.qt.H"
#include "NeovisionII/NeoAnnotate/AnnotationObject.qt.H"
#include "NeovisionII/NeoAnnotate/AnimationDelegate.qt.H"
#include "NeovisionII/NeoAnnotate/AnnotationObjectMgrDelegate.qt.H"
#include <Qt/QtXml>

//######################################################################
void MainWindow::createMenuBar()
{
  QMenu * fileMenu = menuBar()->addMenu(tr("&File"));


  QAction * createDBEntryAction = fileMenu->addAction(tr("&New Annotation"));
  createDBEntryAction->setShortcut(tr("Ctrl+N"));
  connect(createDBEntryAction, SIGNAL(triggered()), this, SLOT(createDBEntry()));

  QAction * saveToDBAction      = fileMenu->addAction("&Save Annotation");
  saveToDBAction->setShortcut(tr("Ctrl+S"));
  connect(saveToDBAction,      SIGNAL(triggered()), this, SLOT(saveAnnotationToDB()));

  QAction * openFromDBAction      = fileMenu->addAction("&Open Existing Annotation");
  openFromDBAction->setShortcut(tr("Ctrl+O"));
  connect(openFromDBAction,    SIGNAL(triggered()), this, SLOT(openAnnotationFromDB()));

  QAction * preferencesAction   = fileMenu->addAction("Preferences...");
  connect(preferencesAction,    SIGNAL(triggered()), this, SLOT(openPrefsDialog()));

  QMenu * connectionMenu = menuBar()->addMenu(tr("&Connection"));
  
  QAction * connectToDBAction = connectionMenu->addAction("Connect To &Database");
  connect(connectToDBAction, SIGNAL(triggered()), &itsDBManager, SLOT(connectToDb()));

  QAction * chooseAnnotationSourceAction = connectionMenu->addAction("Choose Annotation &Source");
  connect(chooseAnnotationSourceAction, SIGNAL(triggered()), &itsDBManager, SLOT(chooseAnnotationSource()));
}

//######################################################################
void MainWindow::createToolbar()
{
  //Initialize all of the qaction buttons with appropriate icons and tooltips
  //Icons from http://www.visualpharm.com/
  QPushButton * zoomInAction    = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/Zoom-In-icon.png"),"",  this);
  QPushButton * zoomOutAction   = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/Zoom-Out-icon.png"),"", this);
  QPushButton * cursorAction    = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/cursor-arrow.png"),"",  this);
  QPushButton * addVertexAction = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/cursor-add.png"),"",    this); 
  QPushButton * remVertexAction = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/cursor-rem.png"),"",    this);
  //QPushButton * rotateAction    = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/rotate-icon.jpg"),"",   this);

  QPushButton * expandAction    = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/expand.png"),"",   this);
  QPushButton * shrinkAction    = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/shrink.png"),"",   this);
  QPushButton * endTrackAction  = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/end-track.png"),"",   this);

  expandAction->setShortcut(tr("Ctrl+>"));
  shrinkAction->setShortcut(tr("Ctrl+<"));
  endTrackAction->setShortcut(tr("Ctrl+e"));

  zoomInAction->setToolTip("Zoom In");
  zoomOutAction->setToolTip("Zoom Out");
  cursorAction->setToolTip("Edit Objects");
  addVertexAction->setToolTip("Add Vertex");
  remVertexAction->setToolTip("Remove Vertex");

  expandAction->setToolTip("Expand size of selected polygon (Ctrl+>)");
  shrinkAction->setToolTip("Shrink size of selected polygon (Ctrl+<)");
  endTrackAction->setToolTip("Make the current object invisible after this frame (Ctrl+e)");

  zoomInAction->setIconSize(QSize(24,24));
  zoomOutAction->setIconSize(QSize(24,24));
  cursorAction->setIconSize(QSize(24,24));
  addVertexAction->setIconSize(QSize(24,24));
  remVertexAction->setIconSize(QSize(24,24));
  
  expandAction->setIconSize(QSize(24,24));
  shrinkAction->setIconSize(QSize(24,24));
  endTrackAction->setIconSize(QSize(24,24));

  QButtonGroup * cursorGroup = new QButtonGroup(this);
  cursorGroup->addButton(cursorAction);
  cursorGroup->addButton(addVertexAction);
  cursorGroup->addButton(remVertexAction);
  //cursorGroup->addButton(rotateAction);

  cursorGroup->addButton(expandAction);  
  cursorGroup->addButton(shrinkAction);  
  cursorGroup->addButton(endTrackAction);

  cursorAction->setCheckable(true);
  addVertexAction->setCheckable(true);
  remVertexAction->setCheckable(true);
  //rotateAction->setCheckable(true);

  expandAction->setCheckable(true);
  shrinkAction->setCheckable(true);
  endTrackAction->setCheckable(true);

  //Create the opacity slider mechanism
  QHBoxLayout * opacityLayout = new QHBoxLayout(this);
  QLabel  * opacityNameLbl  = new QLabel(this);
  QSlider * opacitySlider  = new QSlider(Qt::Horizontal, this);
  QLabel  * opacityAmtLbl   = new QLabel(this);
  QLabel  * opacityPerLbl   = new QLabel(this);
  connect(opacitySlider, SIGNAL(valueChanged(int)), opacityAmtLbl, SLOT(setNum(int)));
  connect(opacitySlider, SIGNAL(valueChanged(int)), itsObjectManager, SLOT(setOpacity(int)));
  opacitySlider->setValue(50);
  opacitySlider->setRange(0, 100);
  opacityNameLbl->setText("Opacity: ");
  opacityPerLbl->setText("%");
  opacityLayout->addWidget(opacityNameLbl);
  opacityLayout->addWidget(opacitySlider);
  opacityLayout->addWidget(opacityAmtLbl);
  opacityLayout->addWidget(opacityPerLbl);
  QGroupBox * opacityBox    = new QGroupBox(this);
  opacityBox->setMaximumWidth(200);
  opacityBox->setLayout(opacityLayout);

  //Create the toolbar
  QToolBar * toolbar = addToolBar(tr("Edit"));

  //Add all of the action buttons to the toolbar
  toolbar->addWidget(zoomInAction);
  toolbar->addWidget(zoomOutAction);
  toolbar->addSeparator();
  toolbar->addSeparator();
  toolbar->addWidget(cursorAction);
  toolbar->addWidget(addVertexAction);
  toolbar->addWidget(remVertexAction);
  //toolbar->addWidget(rotateAction);
  toolbar->addWidget(expandAction);
  toolbar->addWidget(shrinkAction);
  toolbar->addWidget(endTrackAction);
  toolbar->addSeparator();
  toolbar->addSeparator();
  toolbar->addWidget(opacityBox);
  toolbar->addSeparator();

  //Connect each action's triggered signal to the appropriate action
  connect(zoomInAction,    SIGNAL(clicked()), itsMainDisplay,  SLOT(zoomIn()));
  connect(zoomOutAction,   SIGNAL(clicked()), itsMainDisplay,  SLOT(zoomOut()));
  connect(cursorAction,    SIGNAL(clicked()), itsMainDisplay,  SLOT(setActionMode_Cursor()));
  connect(addVertexAction, SIGNAL(clicked()), itsMainDisplay,  SLOT(setActionMode_AddVertex()));
  connect(remVertexAction, SIGNAL(clicked()), itsMainDisplay,  SLOT(setActionMode_RemVertex()));
  //connect(rotateAction,    SIGNAL(clicked()), itsMainDisplay,  SLOT(setActionMode_Rotate()));

  connect(expandAction,    SIGNAL(clicked()), itsObjectManager,  SLOT(expandPolygon()));
  connect(shrinkAction,    SIGNAL(clicked()), itsObjectManager,  SLOT(shrinkPolygon()));
  connect(endTrackAction,  SIGNAL(clicked()), itsObjectManager,  SLOT(endTrack()));

  //Trigger the default cursor action
  cursorAction->click();
}

//######################################################################
QWidget* MainWindow::createObjectList()
{
  //Create a layout for the object list and associated controls
  QTableView *objectList = new QTableView;
  objectList->setSelectionBehavior(QAbstractItemView::SelectRows);
  objectList->setSelectionMode(QAbstractItemView::SingleSelection);
  objectList->setModel(itsObjectManager);
  QVBoxLayout *objectListLayout = new QVBoxLayout;
  objectListLayout->addWidget(objectList);

  AnnotationObjectMgrDelegate* objMgrDel = new AnnotationObjectMgrDelegate(this);
  objMgrDel->setObjCategories(itsDBManager.getObjCategories());
  objectList->setItemDelegate(objMgrDel);

  //When a user clicks on an object in the list, the manager should highlight it, etc.
  connect(objectList, SIGNAL(clicked(const QModelIndex &)), itsObjectManager, SLOT(select(const QModelIndex &)));

  //When the manager selects an object, the object list should highlight the appropriate row
  connect(itsObjectManager, SIGNAL(selectingObject(int)), objectList, SLOT(selectRow(int)));

  QHBoxLayout *objectButtonLayout = new QHBoxLayout;

  QPushButton *addObjectButton = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/Add-icon.png"),"");
  QPushButton *delObjectButton = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/Remove-icon.png"),"");

  addObjectButton->setShortcut(tr("Ctrl+p"));

  connect(addObjectButton, SIGNAL(clicked()), this, SLOT(addObject()));
  connect(delObjectButton, SIGNAL(clicked()), itsObjectManager, SLOT(removeObject()));

  objectButtonLayout->addWidget(addObjectButton);
  objectButtonLayout->addWidget(delObjectButton);
  objectButtonLayout->addStretch();

  objectListLayout->addLayout(objectButtonLayout);
  QWidget *objectListWidget = new QWidget;
  objectListWidget->setLayout(objectListLayout);

  return objectListWidget;
}

//######################################################################
QTableView* MainWindow::createAnimationControls()
{
  itsAnimationView = new QTableView;
  itsAnimationView->setMouseTracking(true);
  itsAnimationView->setSelectionMode(QAbstractItemView::ExtendedSelection);
  itsAnimationView->setSelectionBehavior(QAbstractItemView::SelectItems);
  itsAnimationView->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
  itsAnimationView->horizontalHeader()->setResizeMode(QHeaderView::Fixed);
  itsAnimationView->horizontalHeader()->setDefaultSectionSize(10);
  itsAnimationView->horizontalHeader()->hide();

  itsAnimationView->verticalHeader()->setResizeMode(QHeaderView::Fixed);
  itsAnimationView->verticalHeader()->setDefaultSectionSize(25);

  AnimationDelegate *delegate = new AnimationDelegate(this);
  itsAnimationView->setItemDelegate(delegate);

  itsAnimationView->setDragDropMode(QAbstractItemView::DragDrop);
  itsAnimationView->setDragEnabled(true);

  itsAnimationView->viewport()->installEventFilter(this);

  itsAnimationView->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(itsAnimationView, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(animationViewPopup(const QPoint &)));

  //When the frame changes, the animation view should show a column as selected
  connect(this, SIGNAL(frameIndexChanged(int)), delegate, SLOT(frameIndexChanged(int)));

  //When a column is selected, the animation frame should change
  connect(itsAnimationView, SIGNAL(clicked(const QModelIndex &)), this, SLOT(animationFrameSelected(const QModelIndex &)));

  return itsAnimationView;
}

//######################################################################
MainWindow::MainWindow() :
  itsDBManager(this),
  itsCachedFrameLoader(new CachedFrameLoader),
  itsFramerate(10.0),
  itsPrefsDialog(this),
  itsSettings(new QSettings("iLab", "NeoAnnotate"))
{
  itsDBManager.updateSettings(itsSettings);

  connect(&itsDBManager, SIGNAL(openVideo(QString)), this, SLOT(openVideo(QString)));

  statusBar()->insertPermanentWidget(0, &itsDBStatusLabel);
  statusBar()->insertPermanentWidget(1, new QLabel(" | "));
  statusBar()->insertPermanentWidget(2, &itsAnnotatorLabel);
  itsDBStatusLabel.setText(tr("Not Connected To DB"));
  itsAnnotatorLabel.setText(tr("No Annotator Selected"));

  itsDBManager.connectToDb();
  while(itsDBManager.isConnected() == false)
  {
    QMessageBox msgBox(QMessageBox::Critical, "Not Connected To Database",
        "You must connect to a database to use this application",
        QMessageBox::Abort | QMessageBox::Retry);
    if(msgBox.exec() == QMessageBox::Retry)
    {
      itsDBManager.connectToDb();
    }
    else
    {
      qDebug() << "Exiting. ";
      QCoreApplication::exit(0);
      exit(0);
    }
  }

  buildWindow();
}

//######################################################################
void MainWindow::setDBStatusLabel(QString text)
{
  itsDBStatusLabel.setText(text);
}

//######################################################################
void MainWindow::setAnnotatorLabel(QString text)
{
  itsAnnotatorLabel.setText(text);
}

//######################################################################
void MainWindow::createTimeline()
{
  //Construct the new timeline using the calculated duration
  itsTimeline = new QTimeLine(1, this);

  itsTimeline->setCurveShape(QTimeLine::LinearCurve);
}

//######################################################################
QGroupBox* MainWindow::createTransport()
{
  //Clicking the push button will start the progress bar animation
  itsPlayButton = new QPushButton(QIcon("src/NeovisionII/NeoAnnotate/icons/Play-icon.png"),"");
  connect(itsPlayButton, SIGNAL(clicked()), this, SLOT(playPushed()));

  //Create a slider to show/modify the current frame
  itsProgressBar = new QSlider(this);
  itsProgressBar->setOrientation(Qt::Horizontal);
  itsProgressBar->setRange(0,0);

  connect(itsProgressBar, SIGNAL(sliderMoved(int)), this, SLOT(changeTime(int)));
  connect(itsProgressBar, SIGNAL(sliderPressed()), this, SLOT(sliderPressed()));
  connect(itsProgressBar, SIGNAL(sliderReleased()), this, SLOT(sliderReleased()));

  itsFrameLabel = new QLabel(this);
  itsFrameLabel->setFrameStyle(QFrame::Panel | QFrame::Sunken);
  itsFrameLabel->setMaximumSize(90, 30);
  itsFrameLabel->setMinimumSize(90, 30);

  //Create a layout to put the start button
  //and slider next to each other
  QHBoxLayout *transportLayout = new QHBoxLayout;
  transportLayout->addWidget(itsPlayButton);
  transportLayout->addWidget(itsProgressBar);
  transportLayout->addWidget(itsFrameLabel);

  //Create a group box to hold this layout
  QGroupBox *transportBox = new QGroupBox("Transport");
  transportBox->setLayout(transportLayout);
  transportBox->setFlat(false);

  return transportBox;
}


//######################################################################
void MainWindow::buildWindow()
{
  itsMainDisplay   = new MainDisplay(this);
  itsObjectManager = new AnnotationObjectManager(this);

  connect(itsMainDisplay, SIGNAL(addVertex(QPointF)),    itsObjectManager, SLOT(addVertex(QPointF)));
  connect(itsMainDisplay, SIGNAL(removeVertex(QPointF)), itsObjectManager, SLOT(removeVertex(QPointF)));
  connect(itsMainDisplay, SIGNAL(stretchPolygon(QPointF)), itsObjectManager, SLOT(stretchPolygon(QPointF)));
  createMenuBar();
  createTimeline();
  createToolbar();
  QGroupBox* transport = createTransport();
  QWidget* objectList  = createObjectList();

  //Create the animation controls
  QTableView* animationControls = createAnimationControls();
  itsObjectManager->setAnimationView(animationControls);

  //Create a new layout/widget to hold the object list and transport box
  QVBoxLayout * bottomToolsLayout = new QVBoxLayout;
  bottomToolsLayout->addWidget(animationControls);

  bottomToolsLayout->addWidget(transport);
  QWidget* bottomToolsWidget = new QWidget;
  bottomToolsWidget->setLayout(bottomToolsLayout);

  QDockWidget * ObjectListToolbar = new QDockWidget("Object List");
  ObjectListToolbar->setWidget(objectList);
  ObjectListToolbar->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
  addDockWidget(Qt::LeftDockWidgetArea, ObjectListToolbar);

  //Create a splitter to split the window between the main display, and the controls
  QSplitter *mainSplitter = new QSplitter(this);
  mainSplitter->setOrientation(Qt::Vertical);
  mainSplitter->addWidget(itsMainDisplay);
  mainSplitter->addWidget(bottomToolsWidget);
  QList<int> sizesList;
  sizesList.push_back(400);
  sizesList.push_back(100);
  mainSplitter->setSizes(sizesList);

  //Set up this splitter as the only item in the main layout
  QVBoxLayout* mainLayout = new QVBoxLayout;
  mainLayout->addWidget(mainSplitter);

  //Perform some kind of Qt magic to display the main layout
  QWidget *centralWidget = new QWidget;
  setCentralWidget(centralWidget);
  centralWidget->setLayout(mainLayout);

  //Connect the play button
  connect(itsTimeline, SIGNAL(frameChanged(int)), itsProgressBar, SLOT(setValue(int)));
  connect(itsTimeline, SIGNAL(frameChanged(int)), this, SLOT(updateFrame(int)));
  connect(this, SIGNAL(pausePlayback(bool)), itsTimeline, SLOT(setPaused(bool)));

  setWindowTitle(tr("NeoVision II Video Annotation Tool"));

  //Grab and paint the first frame
  //updateFrame(itsCachedFrameLoader->getFrameRange().getFirst());
}

////////////////////////////////////////////////////////////////////////
// SLOTS
////////////////////////////////////////////////////////////////////////

//######################################################################
void MainWindow::updateFrame(int frameNum)
{
  itsCurrentFrame = frameNum;
  itsFrameLabel->setText(QString("Frame: %1").arg(frameNum, 3, 10, QChar(' ')));
  itsMainDisplay->setImage(itsCachedFrameLoader->getFrame(frameNum));

  emit(frameIndexChanged(frameNum-itsCachedFrameLoader->getFrameRange().getFirst()));

  // Scroll the animation view so that the current frame is always in the middle

  if(itsTimeline->state() == QTimeLine::Running)
	{
		QModelIndex root = itsAnimationView->indexAt(QPoint(0,0));
		QModelIndex currIndex = root.child(1, frameNum);
		itsAnimationView->scrollTo(currIndex, QAbstractItemView::PositionAtCenter);
		itsAnimationView->scrollToTop();
	}
}
   

//######################################################################
void MainWindow::addObject()
{
  //Find the center of the current viewport
  QTransform t = itsMainDisplay->viewportTransform();

  float w = itsMainDisplay->width();
  float h = itsMainDisplay->height();

  float sx = t.m11();
  float sy = t.m22();
  float tx = -t.m31();
  float ty = -t.m32();

  QPointF center_view((w/2.0)/sx, (h/2.0)/sy);
  QPointF topLeft(tx/sx, ty/sy);
  QPointF center = center_view + topLeft;

  //Create a new AnnotationObject, and put it at the center of the viewport,
  //starting at the current frame
  AnnotationObject * obj = new AnnotationObject(
      itsCurrentFrame,
      itsCachedFrameLoader->getFrameRange(),
      center
      );

  obj->setPos(center);

  //Let the object know about the current frame
  obj->frameChanged(itsTimeline->currentFrame());

  itsObjectManager->addObject(obj);
  itsMainDisplay->addObject(obj);

  connect(itsTimeline, SIGNAL(frameChanged(int)), obj, SLOT(frameChanged(int)));
}

void MainWindow::playPushed()
{
  if(itsTimeline->state() == QTimeLine::NotRunning)

  {
    itsPlayButton->setDown(true);
    itsTimeline->start();
  }
  else if(itsTimeline->state() == QTimeLine::Paused)
  {
    itsPlayButton->setDown(true);
    itsTimeline->resume();
  }
  else if(itsTimeline->state() == QTimeLine::Running)
  {
    itsPlayButton->setDown(false);
    itsTimeline->setPaused(true);
  }
}

void MainWindow::changeTime(int frameNum)
{
  //Convert from frame number to milliseconds
  int ms = 1.0/itsFramerate * qreal(frameNum - itsCachedFrameLoader->getFrameRange().getFirst()) * 1000.0;

  itsTimeline->setCurrentTime(ms);
}

void MainWindow::sliderPressed()
{
  if(itsTimeline->state() == QTimeLine::Running)
  {
    timelineWasRunning = true;
    itsTimeline->setPaused(true);
  }
  else
  {
    timelineWasRunning = false;
  }
}

void MainWindow::sliderReleased()
{
  if(timelineWasRunning)
  {
    //itsTimeline->setPaused(false);
    itsTimeline->start();
  }
}


void MainWindow::animationViewPopup(const QPoint & pos)
{
  QPoint globalPos = itsAnimationView->viewport()->mapToGlobal(pos);
  int column = itsAnimationView->indexAt(pos).column();
  int row    = itsAnimationView->indexAt(pos).row();

  itsObjectManager->constructAnimationContextMenu(globalPos, row, column);
}

bool MainWindow::eventFilter(QObject * watched, QEvent * event)
{
  if(watched == itsAnimationView->viewport())
  {
    if(event->type() == QEvent::MouseButtonPress)
    {
      //The user has pressed the mouse button over the animation view
      //Grab the index that was clicked on
      QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
      QPoint pos = mouseEvent->pos();
      QModelIndex index = itsAnimationView->indexAt(pos);
        
      //Change the framenumber
      int frameNumber = index.column() + itsCachedFrameLoader->getFrameRange().getFirst();
      this->changeTime(frameNumber);
        
      //Inform the manager that a cell was clicked on so that it can highlight
      //the proper row, and inform the animationmodel of the click in case
      //there is a drag and drop
      itsObjectManager->setLastAnimViewClick(index);
        
      return false;
    }
    else if(event->type() == QEvent::MouseButtonRelease)
    {
      //The user has released the mouse button over the animation view

      //Grab the index that was clicked on
      QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
      QPoint pos = mouseEvent->pos();
      int column = itsAnimationView->indexAt(pos).column();
      int frameNumber = column + itsCachedFrameLoader->getFrameRange().getFirst();

      //Change the frame number
      this->changeTime(frameNumber);

      return false;
    }
    else if(event->type() == QEvent::KeyPress)
    {

    }
  }
  return false;
}

void MainWindow::animationFrameSelected(const QModelIndex & index)
{
  int frameNumber = index.column() + itsCachedFrameLoader->getFrameRange().getFirst();
  updateFrame(frameNumber);
  changeTime(frameNumber);
}

//void MainWindow::saveAnnotation()
//{
//  //Get the filename from a save file dialog
//  //QString filename = QFileDialog::getSaveFileName(this, "Export Annotation as XML", "", "XML (*.xml)");
//  QFileDialog dialog(this, "Choose a directory");
//  dialog.setFileMode(QFileDialog::DirectoryOnly);
//  if(dialog.exec())
//  {
//    QString directoryName = dialog.selectedFiles()[0];
//
//    std::map<int, std::map<int, AnnotationObjectFrame > > animation = itsObjectManager->renderAnimations();
//
//    saveAnnotationToXML(directoryName, animation);
//  }
//}

//void MainWindow::loadAnnotation()
//{
//  LINFO("Loading...");
//  QFileDialog dialog(this, "Choose a directory");
//  dialog.setFileMode(QFileDialog::DirectoryOnly);
//  if(dialog.exec())
//  {
//    QString directoryName = dialog.selectedFiles()[0];
//    qDebug() << "Directory Name: " << directoryName;
//
//    loadAnnotationFromXML(directoryName);
//  }
//}

void MainWindow::openVideo(QString fileName)
{
  if(fileName == "") return;

  itsFileName = fileName;
  itsCachedFrameLoader->loadVideo(itsFileName);

  updateFrame(0);

  //Grab the frame range from the framegrabber
  FrameRange frameRange = itsCachedFrameLoader->getFrameRange();

  //Compute the total number of frames
  int numFrames = frameRange.getLast() - frameRange.getFirst() - 1;

  //Compute the total duration of the movie
  qreal duration_s = 1.0/(itsFramerate / qreal(numFrames));
  qreal duration_ms = 1000.0 * duration_s;

  itsTimeline->setDuration(duration_ms);

  itsNumFrames = numFrames;

  //Set the timelines first and last frames
  itsTimeline->setFrameRange(frameRange.getFirst(),frameRange.getLast()-1);

  itsProgressBar->setRange(itsTimeline->startFrame(),itsTimeline->endFrame());
}

void MainWindow::openVideo()
{
  QFileDialog dialog(this, tr("Choose a .mgzJ file"));
  dialog.setFileMode(QFileDialog::ExistingFile);
  dialog.setNameFilter(tr("mgzJ Files (*.mgzJ)"));

  if(dialog.exec())
  {
     openVideo(dialog.selectedFiles()[0]);
  }
}

/*
//TODO: This should be moved out of here into it's own class which inherits from some kind of general state importer/exporter class.
//TODO: Implement a compact binary output for this data in addition to the bulky xml
void MainWindow::saveAnnotationToXML(QString directoryName, std::map<int, std::map<int, AnnotationObjectFrame > > animation)
{
  std::vector<QString> frame_filenames;

  std::map<int, std::map<int, AnnotationObjectFrame> >::iterator animIt;

  for(animIt = animation.begin(); animIt != animation.end(); animIt++)
  {

    int fnum = animIt->first;

    //Open the file for this frame
    QString filename = directoryName + QString("/frame%1.xml").arg(fnum);

    QString frame_filename = QString("frame%1").arg(fnum); 
    
    QDomDocument frameDoc("frame");

    QDomElement node_scenes = frameDoc.createElement("scenes");
    frameDoc.appendChild(node_scenes);

    QDomElement node_annotation = frameDoc.createElement("annotation");
    node_scenes.appendChild(node_annotation);
    
    //TODO: This should be replaced with the real frame filename
    QDomElement node_filename = frameDoc.createElement("filename");
    node_annotation.appendChild(node_filename);
    QDomText node_filename_t = frameDoc.createTextNode(QString("%1").arg(fnum));
    node_filename.appendChild(node_filename_t);

    QDomElement node_framenumber = frameDoc.createElement("framenumber");
    node_annotation.appendChild(node_framenumber);
    QDomText node_framenumber_t = frameDoc.createTextNode(QString("%1").arg(fnum));
    node_framenumber.appendChild(node_framenumber_t);

    //TODO: What's this?
    QDomElement node_source = frameDoc.createElement("source");
    node_annotation.appendChild(node_source);
    QDomText node_source_t = frameDoc.createTextNode("NeoVision");
    node_source.appendChild(node_source_t);

    //Grab all of the objects in this frame
    std::map<int, AnnotationObjectFrame> objects = animIt->second;
    std::map<int, AnnotationObjectFrame>::iterator objectsIt;

    //Loop through all of the objects and put them into the xml tree
    for(objectsIt = objects.begin(); objectsIt != objects.end(); objectsIt++)
    {
      AnnotationObjectFrame f = objectsIt->second;

      QDomElement node_object = frameDoc.createElement("object");
      node_object.setAttribute("vis", f.ObjectFrameState.visible?"1":"0");
      node_object.setAttribute("key", f.ObjectFrameState.is_keyframe?"1":"0");
      node_annotation.appendChild(node_object);

      QDomElement node_name = frameDoc.createElement("name");
      node_object.appendChild(node_name);
      QDomText node_name_t = frameDoc.createTextNode(f.ObjectName);
      node_name.appendChild(node_name_t);

      QDomElement node_id = frameDoc.createElement("id");
      node_object.appendChild(node_id);
      QDomText node_id_t = frameDoc.createTextNode(QString("%1").arg(f.ObjectId));
      node_id.appendChild(node_id_t);

      QDomElement node_description = frameDoc.createElement("description");
      node_object.appendChild(node_description);
      QDomText node_description_t = frameDoc.createTextNode(f.ObjectType);
      node_description.appendChild(node_description_t);

      //Insert the position of the object into the file.  Note that this is
      //_only_ so that we can reload this file to get an exact representation
      //of the original keyframing. This position is absolute, and all of the
      //vertex positions are absolute as well so that when we reload, we will
      //need to subtract the object position from the vertex positions to make
      //them relative again.
      QDomElement node_x = frameDoc.createElement("x");
      node_object.appendChild(node_x);
      QDomText node_x_t = frameDoc.createTextNode(QString("%1").arg(f.ObjectFrameState.pos.x()));
      node_x.appendChild(node_x_t);
      QDomElement node_y = frameDoc.createElement("y");
      node_object.appendChild(node_y);
      QDomText node_y_t = frameDoc.createTextNode(QString("%1").arg(f.ObjectFrameState.pos.y()));
      node_y.appendChild(node_y_t);

      QDomElement node_polygon = frameDoc.createElement("polygon");
      node_object.appendChild(node_polygon);

      //Loop through all of the vertices and append them to the object branch
      std::map<int, ObjectAnimation::FrameState>::iterator vIt;
      for(vIt = f.VertexFrames.begin(); vIt != f.VertexFrames.end(); vIt++)
      {
        ObjectAnimation::FrameState vState = vIt->second;
        QDomElement node_pt = frameDoc.createElement("pt");
        node_pt.setAttribute("vis", vState.visible?"1":"0");
        node_pt.setAttribute("key", vState.is_keyframe?"1":"0");
        node_polygon.appendChild(node_pt);

        QDomElement node_x = frameDoc.createElement("x");
        node_pt.appendChild(node_x);
        QDomText node_x_t = frameDoc.createTextNode(QString("%1").arg(vState.pos.x()));
        node_x.appendChild(node_x_t);

        QDomElement node_y = frameDoc.createElement("y");
        node_pt.appendChild(node_y);
        QDomText node_y_t = frameDoc.createTextNode(QString("%1").arg(vState.pos.y()));
        node_y.appendChild(node_y_t);

        QDomElement node_id = frameDoc.createElement("id");
        node_pt.appendChild(node_id);
        QDomText node_id_t = frameDoc.createTextNode(QString("%1").arg(vIt->first));
        node_id.appendChild(node_id_t);
      }
    }

    //Write the file to disk
    QString fullFrameFilename = directoryName + "/" + frame_filename + ".xml";
    QFile frame_file(fullFrameFilename);
    frame_filenames.push_back(fullFrameFilename);
    if(!frame_file.open(QIODevice::WriteOnly | QIODevice::Text))
    { 
      qDebug() << "Couldn't write " << fullFrameFilename << "!";
      return; 
    }
    QTextStream out(&frame_file);
    out << frameDoc.toString();
  }


  QDomDocument indexDoc("index");

  QDomElement node_scenes = indexDoc.createElement("scenes");
  indexDoc.appendChild(node_scenes);

  std::vector<QString>::iterator includeIt;
  for(includeIt = frame_filenames.begin(); includeIt != frame_filenames.end(); includeIt++)
  {
    QDomElement node_include = indexDoc.createElement("include");
    node_include.setAttribute("filename", *includeIt);
    node_scenes.appendChild(node_include);
  }

  //Write the file to disk
  QString indexFilename = directoryName + "/index.xml";
  QFile index_file(indexFilename);
  if(!index_file.open(QIODevice::WriteOnly | QIODevice::Text))
  { 
    qDebug() << "Couldn't write " << indexFilename << "!";
    return; 
  }
  QTextStream out(&index_file);
  out << indexDoc.toString();
}


void MainWindow::loadAnnotationFromXML(QString Directory)
{
  QString indexFileName = Directory + "/index.xml";
  QDomDocument indexDoc("index");
  QFile file(indexFileName);
  if(!file.open(QIODevice::ReadOnly))
  {
    LINFO("Could Not Open %s", indexFileName.toStdString().c_str());
    return;
  }
  if(!indexDoc.setContent(&file))
  {
    LINFO("Could Not Open %s - Bad File", indexFileName.toStdString().c_str());
    file.close();
    return;
  }
  file.close();

  //Load the document, and grab the scenes branch
  QDomElement indexDocElem = indexDoc.documentElement();
  QDomNode node_scenes = indexDoc.elementsByTagName("scenes").item(0);
  if(node_scenes.isNull())
  {
    LINFO("Error - malformed index.xml file! Line: %d", __LINE__);
    return;
  }

  //Clear the Annotation Object Manager
  itsObjectManager->clear();

  //Get the frame range from the framegrabber so that we can only load
  //animation for the relavent frames.
  FrameRange videoFrameRange = itsCachedFrameLoader->getFrameRange();

  //Loop through each include node (frame number) in the scenes
  //TODO: Parse out the frame number and just continue if we are out of the frame range
  QDomNodeList fileList = node_scenes.childNodes();
  for(int fileIdx = 0; fileIdx < fileList.size(); fileIdx++)
  {
    //Load the XML file included in the index
    QDomNode node_file = fileList.at(fileIdx);
    QString frameFileName = node_file.attributes().namedItem("filename").toAttr().value();
    QDomDocument frameDoc("frame");
    QFile file(frameFileName);
    if(!file.open(QIODevice::ReadOnly))
    {
      LINFO("Could Not Open %s", frameFileName.toStdString().c_str());
      return;
    }
    if(!frameDoc.setContent(&file))
    {
      LINFO("Could Not Open %s - Bad File", frameFileName.toStdString().c_str());
      file.close();
      return;
    }
    file.close();

    //Load the document
    QDomElement frameDocElem = frameDoc.documentElement();

    //Find the framenumber for this document
    QDomNode node_framenumber = frameDocElem.elementsByTagName("framenumber").item(0);
    if(node_framenumber.isNull())
    { LINFO("Error - malformed xml file! Line: %d", __LINE__); return; }
    int fnum = node_framenumber.firstChild().nodeValue().toInt();

    //Loop through all of the objects in the document
    QDomNodeList objects = frameDocElem.elementsByTagName("object");
    for(int objIdx=0; objIdx<objects.size(); objIdx++)
    {
      QDomNode node_object = objects.at(objIdx);

      //Grab the object's ID
      int objectId = node_object.firstChildElement("id").firstChild().nodeValue().toInt();

      //Find out if this frame is a keyframe for this object
      bool object_isKey = node_object.attributes().namedItem("key").toAttr().value().toInt();

      //Find out if the object is visible in this frame
      bool object_isVis = node_object.attributes().namedItem("vis").toAttr().value().toInt();
      
      //Grab the object's position
      QPointF objectPos;
      objectPos.setX(node_object.firstChildElement("x").firstChild().nodeValue().toDouble());
      objectPos.setY(node_object.firstChildElement("y").firstChild().nodeValue().toDouble());
      
      //Read the object's vertices' states and store them in VertexFrames
      QDomNode node_vertex = node_object.firstChildElement("polygon").firstChildElement("pt");
      std::map<int, ObjectAnimation::FrameState> VertexFrames;
      while(!node_vertex.isNull())
      {
        ObjectAnimation::FrameState vertexState;
        vertexState.pos.setX(node_vertex.firstChildElement("x").firstChild().nodeValue().toDouble());
        vertexState.pos.setY(node_vertex.firstChildElement("y").firstChild().nodeValue().toDouble());
        vertexState.pos -= objectPos;
        vertexState.visible = node_vertex.attributes().namedItem("vis").toAttr().value().toInt();
        vertexState.is_keyframe = node_vertex.attributes().namedItem("key").toAttr().value().toInt();

        int vertexId = node_vertex.firstChildElement("id").firstChild().nodeValue().toDouble();
        VertexFrames[vertexId] = vertexState;

        node_vertex = node_vertex.nextSibling();
      }

      //Check to see if the object already exists in the database
      AnnotationObject* obj = itsObjectManager->getObjectById(objectId);
      if(obj == NULL)
      {
        //If the object is not in the database, and we have found it's first keyframe then we should
        //create it and insert it into storage.

        QString name        = node_object.firstChildElement("name").firstChild().nodeValue();
        int category        = node_object.firstChildElement("description").firstChild().nodeValue().toInt();

        //Create a new object and insert it into the Object Manager
        obj =
          new AnnotationObject(
              fnum,
              videoFrameRange,
              objectPos, name, category
              );
        obj->forceId(objectId);
        obj->clearAnimation();
        obj->setVertices(VertexFrames);

        itsObjectManager->addObject(obj);
        itsMainDisplay->addObject(obj);
        connect(itsTimeline, SIGNAL(frameChanged(int)), obj, SLOT(frameChanged(int)));
      }

      //Check to see if we have a new keyframe for our object
      if(object_isKey)
      {
        obj->setKeyframe(fnum, objectPos, object_isVis);
      }

      //Roll through all of the vertices to see if they have any keyframes at this frame
      std::map<int, ObjectAnimation::FrameState>::iterator vIt;
      for(vIt=VertexFrames.begin(); vIt!=VertexFrames.end(); vIt++)
      {
        ObjectAnimation::FrameState vertexState = vIt->second;
        if(vertexState.is_keyframe)
        {
          AnnotationObjectVertex* vertex = obj->getVertexById(vIt->first);
          vertex->setKeyframe(fnum, vertexState.pos, vertexState.visible);
        }
      }

    }
  }

  LINFO("Finished Import");
}
*/

void MainWindow::saveAnnotationToDB()
{
  itsDBManager.saveAnnotation();
}

void MainWindow::openAnnotationFromDB()
{
  itsDBManager.openAnnotation();
}

void MainWindow::createDBEntry()
{
  itsDBManager.createDBEntry();
}

void MainWindow::openPrefsDialog()
{

  if(itsSettings->contains("archiveLoc"))
    itsPrefsDialog.archiveLocEdit->setText(itsSettings->value("archiveLoc").toString());
  if(itsSettings->contains("workingLoc"))
    itsPrefsDialog.workingLocEdit->setText(itsSettings->value("workingLoc").toString());
  if(itsSettings->contains("incomingLoc"))
    itsPrefsDialog.incomingLocEdit->setText(itsSettings->value("incomingLoc").toString());

  if(itsPrefsDialog.exec())
  {
    QString archiveLoc  = itsPrefsDialog.archiveLocEdit->text();
    QString workingLoc  = itsPrefsDialog.workingLocEdit->text();
    QString incomingLoc = itsPrefsDialog.incomingLocEdit->text();

    itsSettings->setValue("archiveLoc",  archiveLoc);
    itsSettings->setValue("workingLoc",  workingLoc);
    itsSettings->setValue("incomingLoc", incomingLoc);

  }
}
#endif //MAINWINDOW_C

