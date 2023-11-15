#include "NeovisionII/NeoAnnotate/DBManager.qt.H"
#include "NeovisionII/NeoAnnotate/MainWindow.qt.H"
#include <Qt/QtCore>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QtSql/QSqlQuery>
#include <Qt3Support/q3urloperator.h>
#include <Qt3Support/q3network.h>
#include "NeovisionII/NeoAnnotate/AnnotationObjectManager.qt.H"
#include "NeovisionII/NeoAnnotate/AnnotationObject.qt.H"
#include <set>

// ######################################################################
ConnectionDialog::ConnectionDialog(QWidget* parent)
  : QDialog(parent)
{ 
  QFormLayout *layout = new QFormLayout;

  serverNameEdit = new QLineEdit("isvn.usc.edu");
  layout->addRow(tr("&Server Name"), serverNameEdit);

  dbNameEdit = new QLineEdit("neo2annotations");
  layout->addRow(tr("&Database Name"), dbNameEdit);

  userNameEdit = new QLineEdit("neo2");
  layout->addRow(tr("&User Name"), userNameEdit);
    
  passwordEdit = new QLineEdit("neo2!");
  passwordEdit->setEchoMode(QLineEdit::Password);
  layout->addRow(tr("&Password"), passwordEdit);

  QPushButton* cancelButton = new QPushButton("&Cancel", this);
  connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
  layout->addRow(cancelButton);

  QPushButton* connectButton = new QPushButton("&Connect", this);
  connect(connectButton, SIGNAL(clicked()), this, SLOT(accept()));
  layout->addRow(connectButton);

  setLayout(layout);
}

// ######################################################################
NewDBEntryDialog::NewDBEntryDialog(QWidget* parent, DBManager *mgr, QSqlDatabase* _db)
  : QDialog(parent),
    itsDBManager(mgr),
    db(_db),
    itsSceneUid("-1")
{
  setMinimumSize(QSize(500, 200));
  QFormLayout *layout = new QFormLayout;

  labComboBox = new QComboBox(this);
  layout->addRow(tr("Lab"), labComboBox);
  connect(labComboBox, SIGNAL(activated(int)), this, SLOT(fillInCameraOperator(int)));

  QHBoxLayout* fileChooserLayout = new QHBoxLayout;
  fileNameLabel                  = new QLabel;
  QPushButton* fileNameButton    = new QPushButton("Browse...");
  connect(fileNameButton, SIGNAL(clicked()), this, SLOT(browse()));
  fileChooserLayout->addWidget(fileNameLabel);
  fileChooserLayout->addWidget(fileNameButton);
  layout->addRow(tr("FileName"), fileChooserLayout);

  cameraComboBox = new QComboBox(this);
  layout->addRow(tr("Camera"), cameraComboBox);

  startDateTimeEdit = new QDateTimeEdit(this);
  layout->addRow(tr("Start Time"), startDateTimeEdit);

  endDateTimeEdit = new QDateTimeEdit(this);
  layout->addRow(tr("End Time"), endDateTimeEdit);

  timeZoneBox = new QComboBox(this);
  timeZoneBox->addItem("NST	  Newfoundland Standard Time",    "NST");	  
  timeZoneBox->addItem("NDT	  Newfoundland Daylight Time",    "NDT");	  
  timeZoneBox->addItem("AST	  Atlantic Standard Time",        "AST");	  
  timeZoneBox->addItem("ADT	  Atlantic Daylight Time",        "ADT");	  
  timeZoneBox->addItem("EST	  Eastern Standard Time",         "EST");	  
  timeZoneBox->addItem("EDT	  Eastern Daylight Time",         "EDT");	  
  timeZoneBox->addItem("CST	  Central Standard Time",         "CST");	  
  timeZoneBox->addItem("CDT	  Central Daylight Time",         "CDT");	  
  timeZoneBox->addItem("MST	  Mountain Standard Time",        "MST");	  
  timeZoneBox->addItem("MDT	  Mountain Daylight Time",        "MDT");	  
  timeZoneBox->addItem("PST	  Pacific Standard Time",         "PST");	  
  timeZoneBox->addItem("PDT	  Pacific Daylight Time",         "PDT");	  
  timeZoneBox->addItem("AKST	Alaska Standard Time",          "AKST");	
  timeZoneBox->addItem("AKDT	Alaska Daylight Time",          "AKDT");	
  timeZoneBox->addItem("HAST	Hawaii-Aleutian Standard Time", "HAST");	
  timeZoneBox->addItem("HADT	Hawaii-Aleutian Daylight Time", "HADT");	
  layout->addRow(tr("Time Zone"), timeZoneBox);

  operatorComboBox = new QComboBox(this);
  layout->addRow(tr("Operator"), operatorComboBox);

  weatherComboBox = new QComboBox(this);
  layout->addRow(tr("Weather"), weatherComboBox);

  startFrameEdit   = new QLineEdit(this);
  startFrameEdit->setValidator(new QIntValidator(0, 9999999, this));
  layout->addRow(tr("Start Frame"), startFrameEdit);

  numFramesEdit   = new QLineEdit(this);
  numFramesEdit->setValidator(new QIntValidator(0, 9999999, this));
  layout->addRow(tr("Number of Frames"), numFramesEdit);

  frameRateEdit = new QLineEdit(this);
  frameRateEdit->setValidator(new QIntValidator(1, 300, this));
  layout->addRow(tr("Frame Rate"), frameRateEdit);

  sceneNameEdit = new QLineEdit(this);
  layout->addRow(tr("Scene Name"), sceneNameEdit);

  QPushButton* createButton = new QPushButton("&Create", this);
  connect(createButton, SIGNAL(clicked()), this, SLOT(commitNewScene()));
  layout->addRow(createButton);

  QPushButton* cancelButton = new QPushButton("&Cancel", this);
  connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
  layout->addRow(cancelButton);

  setLayout(layout);
}

// ######################################################################
void NewDBEntryDialog::browse()
{
  QString fileName = QFileDialog::getOpenFileName(this, 
      tr("Select Video File"), QString(VIDEO_INCOMING_LOCATION), tr("Video Files (*.MTS *.AVI *.MPG *.MPEG)"));
  QFileInfo inputFileInfo(fileName);
  QFileInfo mgzJFileInfo(inputFileInfo.dir().path() + "/" + inputFileInfo.baseName() + ".mgzJ");
  if(!mgzJFileInfo.exists())
  {
    QMessageBox msgBox(QMessageBox::Critical, "No mgzJ File Found",
        "The input file you selected does not have a corresponding .mgzJ file in the same directory.\n"
        "Please create a .mgzJ file for this video, and try again.",
        QMessageBox::Ok);
    msgBox.exec();
  }
  else
  {
    fileNameLabel->setText(fileName);
  }
}

// ######################################################################
void NewDBEntryDialog::commitNewScene()
{

  QDateTime startTime = startDateTimeEdit->dateTime();
  QDateTime endTime = endDateTimeEdit->dateTime();
  if(startTime >= endTime)
  {
    QMessageBox msgBox(QMessageBox::Critical, "Bad Dates!",
        "Start date/time cannot be after end date/time!",
        QMessageBox::Ok);
    msgBox.exec();
  }
  else
  {
    QString timeZone  = timeZoneBox->itemData(timeZoneBox->currentIndex()).toString();
    QString startDateTime = startTime.toString("yyyy-MM-dd hh:mm ") + timeZone;
    QString endDateTime   = endTime.toString("yyyy-MM-dd hh:mm ") + timeZone;

    QString sourceFileName = fileNameLabel->text();
    QString fileSuffix = QFileInfo(sourceFileName).suffix();

    QString insertStatement;
    insertStatement += "INSERT INTO scene ";
    insertStatement += "(starttime, endtime, camera, operator, url, numframes, framerate, weather, startframe, name) ";
    insertStatement += "VALUES (";
    insertStatement += "\'" + startDateTime + "\', ";
    insertStatement += "\'" + endDateTime   + "\', ";
    insertStatement += cameraComboBox->itemData(cameraComboBox->currentIndex()).toString()     + ", ";
    insertStatement += operatorComboBox->itemData(operatorComboBox->currentIndex()).toString() + ", "; 
    insertStatement += "\'ext://" + fileSuffix + "\', ";
    insertStatement += numFramesEdit->text() + ", ";
    insertStatement += frameRateEdit->text() + ", ";
    insertStatement += weatherComboBox->itemData(weatherComboBox->currentIndex()).toString() + ", ";
    insertStatement += startFrameEdit->text() + ", ";
    insertStatement += "\'" + sceneNameEdit->text() + "\'";
    insertStatement += ")";

    db->exec(insertStatement);

    if(db->lastError().isValid())
    {
      QMessageBox msgBox(QMessageBox::Critical, "Database Insert Error",
          "Error inserting into database. Reason:\n" + db->lastError().text() + "\n\n" +
					"Insert Statement Was:\n"+insertStatement,
          QMessageBox::Ok);
      msgBox.exec();
    }
    else
    {
      // Get the most recently obtained sequence value for our uid - this should be safe,
      // even for multiple users
      QSqlQuery query = db->exec("select currval('scene_uid_seq')");

      if(db->lastError().isValid())
      {
        QMessageBox msgBox(QMessageBox::Critical, "Database Select Error",
            "Error retrieving last insertion UID. Reason:\n" + db->lastError().text(),
            QMessageBox::Ok);
        msgBox.exec();
      }
      else
      {
        // Copy the file to it's new home in the neo2data/archive directory, renamed to the fetched uid
        query.next();
        itsSceneUid = query.value(0).toString();

        {
          QString destinationFileName = itsDBManager->itsArchiveLoc + "/" + db->databaseName() + "/" +
            itsSceneUid + "." + fileSuffix;
          QProcess copyProc;
          QString execString("/bin/mv " + sourceFileName + " " + destinationFileName);
          qDebug() << execString;
          QProgressDialog progressDialog(this);
          progressDialog.setAutoClose(false);
          progressDialog.setAutoReset(false);
          progressDialog.setLabel(new QLabel("Copying File, Please Wait...", this));
          progressDialog.setCancelButton(0);
          progressDialog.setRange(0, 0);
          connect(&copyProc, SIGNAL(finished(int, QProcess::ExitStatus)), this,
                  SLOT(copyFinished(int, QProcess::ExitStatus)));
          connect(this, SIGNAL(closeProgressDialog()), &progressDialog, SLOT(cancel()));
          copyProc.start(execString);
          progressDialog.exec();
          copyProc.waitForFinished(-1);
        }
        {
          QFileInfo inputFileInfo(sourceFileName);
          QString sourceMgzJFileName = inputFileInfo.dir().path() + "/" + inputFileInfo.baseName() + ".mgzJ";
          itsMgzJFileName   = itsDBManager->itsWorkingLoc + "/" + db->databaseName() + "/" + itsSceneUid + ".mgzJ";

          qDebug() << "Setting mgzJ filename: " << itsMgzJFileName;

          QProcess copyProc;
          QString execString("/bin/mv " + sourceMgzJFileName + " " + itsMgzJFileName);
          qDebug() << execString;
          QProgressDialog progressDialog(this);
          progressDialog.setAutoClose(false);
          progressDialog.setAutoReset(false);
          progressDialog.setLabel(new QLabel("Copying File, Please Wait...", this));
          progressDialog.setCancelButton(0);
          progressDialog.setRange(0, 0);
          connect(&copyProc, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(copyFinished(int, QProcess::ExitStatus)));
          connect(this, SIGNAL(closeProgressDialog()), &progressDialog, SLOT(cancel()));
          copyProc.start(execString);
          progressDialog.exec();
          copyProc.waitForFinished(-1);
        }
        accept();
      }
    }
  }
}

// ######################################################################
void NewDBEntryDialog::copyFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
  emit closeProgressDialog();
}

// ######################################################################
void NewDBEntryDialog::showEvent(QShowEvent* event)
{

  // Fill in the lab combo box first
  labComboBox->clear();
  QSqlQuery labsQuery = db->exec("SELECT uid, name, institution FROM lab");
  while(labsQuery.next())
  {
    int uid             = labsQuery.value(0).toInt();
    QString name        = labsQuery.value(1).toString();
    QString institution = labsQuery.value(2).toString();
    labComboBox->addItem(name + " @ " + institution, uid);
  }

  QSqlQuery weatherQuery = db->exec("SELECT uid, name FROM weather");
  while(weatherQuery.next())
  {
    int uid             = weatherQuery.value(0).toInt();
    QString name        = weatherQuery.value(1).toString();
    weatherComboBox->addItem(name, uid);
  }

  fillInCameraOperator(0);

  QDialog::showEvent(event);
}

// ######################################################################
void NewDBEntryDialog::fillInCameraOperator(int labComboBoxRow)
{
  int labUID = labComboBox->itemData(labComboBoxRow).toInt();
  cameraComboBox->clear();
  operatorComboBox->clear();

  QSqlQuery cameraQuery =
    db->exec("SELECT uid, name, manufacturer, model FROM camera WHERE lab="+QString::number(labUID));
  while(cameraQuery.next())
  {
    int uid              = cameraQuery.value(0).toInt();
    QString name         = cameraQuery.value(1).toString();
    QString manufacturer = cameraQuery.value(2).toString();
    QString model        = cameraQuery.value(3).toString();
    cameraComboBox->addItem(name + ": " + manufacturer + " " + model, uid);
  }

  QSqlQuery operatorQuery =
    db->exec("SELECT uid, firstname, lastname, jobtitle FROM operator WHERE lab="+QString::number(labUID));
  while(operatorQuery.next())
  {
    int uid            = operatorQuery.value(0).toInt();
    QString firstname  = operatorQuery.value(1).toString();
    QString lastname   = operatorQuery.value(2).toString();
    QString jobtitle   = operatorQuery.value(3).toString();
    operatorComboBox->addItem(lastname + ", " + firstname + " : " + jobtitle, uid);
  }
qDebug() << "Getting Operator!: " <<  db->lastError();
}

// ######################################################################
SelectAnnotationSourceDialog::SelectAnnotationSourceDialog(QWidget* parent, QSqlDatabase* _db) :
  QDialog(parent),
  itsSourceUid("-1"),
  db(_db)
{
  setMinimumSize(QSize(500, 200));
  QVBoxLayout *layout = new QVBoxLayout;

  itsSourceTree = new QTreeWidget(this);
  QStringList headerList;
  headerList << "UID" << "Name" << "Validation Level";
  itsSourceTree->setHeaderLabels(headerList);

  layout->addWidget(itsSourceTree);

  QPushButton* openButton = new QPushButton("&Select", this);
  connect(openButton, SIGNAL(clicked()), this, SLOT(selectSource()));
  layout->addWidget(openButton);

  QPushButton* cancelButton = new QPushButton("&Cancel", this);
  connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
  layout->addWidget(cancelButton);

  setLayout(layout);
}

void SelectAnnotationSourceDialog::showEvent(QShowEvent* event)
{
  itsSourceTree->clear();
  QSqlQuery sourceQuery =
    db->exec("SELECT uid, name, validationlevel FROM annotationsource WHERE categ=2");
  while(sourceQuery.next())
  {
    QString uid   = sourceQuery.value(0).toString();
    QString name  = sourceQuery.value(1).toString();
    QString level = sourceQuery.value(2).toString();
    QStringList source;
    source << uid << name << level;
    itsSourceTree->addTopLevelItem(new QTreeWidgetItem(source));
  }

  QDialog::showEvent(event);
}

// ######################################################################
void SelectAnnotationSourceDialog::selectSource()
{

  QTreeWidgetItem* selectedSource = itsSourceTree->currentItem();
  if(selectedSource == NULL) return;

  itsSourceUid = selectedSource->text(0);

  if(itsSourceUid != "-1")
    static_cast<MainWindow*>(parent())->setAnnotatorLabel("Annotating As " + selectedSource->text(1));
  else
    static_cast<MainWindow*>(parent())->setAnnotatorLabel("No Annotator Selected");

  accept();
}

// ######################################################################
OpenDBEntryDialog::OpenDBEntryDialog(QWidget* parent, DBManager *mgr, QSqlDatabase* _db) :
  QDialog(parent),
  itsDBManager(mgr),
  itsSceneUid("-1"),
  db(_db)
{ 
  setMinimumSize(QSize(500, 200));
  QVBoxLayout *layout = new QVBoxLayout;

  itsSceneTree = new QTreeWidget(this);
  QStringList headerList;
  headerList << "UID" << "Name" << "Operator" << "Camera" << "Start Time" << "End Time";
  itsSceneTree->setHeaderLabels(headerList);
  layout->addWidget(itsSceneTree);

  QPushButton* openButton = new QPushButton("&Open", this);
  connect(openButton, SIGNAL(clicked()), this, SLOT(openEntry()));
  layout->addWidget(openButton);

  QPushButton* cancelButton = new QPushButton("&Cancel", this);
  connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
  layout->addWidget(cancelButton);

  setLayout(layout);
}

void OpenDBEntryDialog::showEvent(QShowEvent* event)
{
  itsSceneTree->clear();

  QSqlQuery sceneQuery =
    db->exec("SELECT uid, name, operator, camera, starttime, endtime FROM scene ORDER BY uid"); 

  while(sceneQuery.next())
  {
    QString uid        = sceneQuery.value(0).toString();
    QString name       = sceneQuery.value(1).toString();
    QString operatorId = sceneQuery.value(2).toString();
    QString cameraId   = sceneQuery.value(3).toString();
    QString startTime  = sceneQuery.value(4).toString();
    QString endTime    = sceneQuery.value(5).toString();

    QSqlQuery cameraQuery =
      db->exec("SELECT name FROM camera WHERE uid="+cameraId); 
    QString cameraName = "";
    if(cameraQuery.size() > 0)
    {
      cameraQuery.next();
      cameraName = cameraQuery.value(0).toString();
    }

    QSqlQuery operatorQuery =
      db->exec("SELECT firstname,lastname FROM operator WHERE uid="+operatorId); 
    QString operatorName = "";
    if(operatorQuery.size() > 0)
    {
      operatorQuery.next();
      operatorName = operatorQuery.value(1).toString() + ", " + operatorQuery.value(0).toString();
    }

    QStringList sceneItem;
    sceneItem << uid << name << operatorName << cameraName << startTime << endTime;
    
    QTreeWidgetItem* newItem = new QTreeWidgetItem(sceneItem);

    QString mgzJFileName = itsDBManager->itsWorkingLoc + "/" + db->databaseName() + "/" + uid + ".mgzJ";
    QFileInfo mgzJFileInfo(mgzJFileName);
    if(mgzJFileInfo.exists() == false)
      newItem->setDisabled(true);

    itsSceneTree->addTopLevelItem(newItem);
  }

  QDialog::showEvent(event);
}

void OpenDBEntryDialog::openEntry()
{
  QTreeWidgetItem* selectedScene = itsSceneTree->currentItem();
  if(selectedScene)
  {
    itsSceneUid = selectedScene->text(0);

    QString mgzJFileName = itsDBManager->itsWorkingLoc + "/" + db->databaseName() + "/" + itsSceneUid + ".mgzJ";
    QFileInfo mgzJFileInfo(mgzJFileName);
    if(mgzJFileInfo.exists())
    {
      itsMgzJFileName = mgzJFileName;
      accept();
    }
    else
    {
      QMessageBox msgBox(QMessageBox::Warning, "Could Not Open MgzJ File",
          "No .mgzJ file found for this scene.\n\n"
          "You must ensure that the cached .mgzJ file (" + mgzJFileName + ") "
          "exists before trying to reopen an annotation.\n\n"
          "**** This error should not have occured! Please file a bug report with the following info:\n"
          "File: " + QString(__FILE__) + " Line: " + QString(__LINE__),
          QMessageBox::Ok);
      msgBox.exec();
    }
  }
  else
  {
    QMessageBox msgBox(QMessageBox::Warning, "No Scene Selected",
        "You did not select a scene!\n\n"
        "If all scenes are greyed out, then you must reconvert some original "
        "footage into the .mgzJ format. Please see Dr. Itti or Rand for "
        "instructions on doing this.",
        QMessageBox::Ok);
    msgBox.exec();
  }
}

// ######################################################################

// ######################################################################
void DBManager::connectToDb()
{
  if(itsConnDialog.exec())
  {
    db = QSqlDatabase::addDatabase("QPSQL");
    db.setHostName(itsConnDialog.serverNameEdit->text());
    db.setDatabaseName(itsConnDialog.dbNameEdit->text());
    db.setUserName(itsConnDialog.userNameEdit->text());
    db.setPassword(itsConnDialog.passwordEdit->text());

    connected = db.open();


    if(!connected)
    {
      QMessageBox msgBox(QMessageBox::Warning, "Could Not Connect",
          "Could Not Connect To The Database.\nReason:" + db.lastError().text(),
          QMessageBox::Ok);
      msgBox.exec();
    }
  }


  if(connected)
  {
    static_cast<MainWindow*>(parent())->setDBStatusLabel("Connected To " + db.hostName());
    chooseAnnotationSource();
  }
  else
    static_cast<MainWindow*>(parent())->setDBStatusLabel("Not Connected To DB");
}

void DBManager::chooseAnnotationSource()
{
  while(!connected) connectToDb();

  while(!itsSelectAnnotationSourceDialog.exec())
  {
      QMessageBox msgBox(QMessageBox::Warning, "No Annotation Source Selected",
          "You must select yourself as an annotation source to save or load any work.",
          QMessageBox::Ok);
      msgBox.exec();
  }
}

// ######################################################################
namespace 
{
  struct PolygonKeyframe
  {
    int scene;
    std::vector<QPointF> vertices;
    std::vector<bool> verticesVisible;
    std::vector<bool> verticesKeyframe;
    int annotationSource;
    int object;
    QPointF pos;
    bool visible;
    bool keyframe;
  };
}
void DBManager::saveAnnotation()
{
  // Make sure we have a valid annotation source, and are connected to the database
  if(!connected) connectToDb();
  if(!connected)
  {
    QMessageBox msgBox(QMessageBox::Warning, "Not connected to a database",
        "Your work has not been saved!\n\n"
        "You must connect to a database before saving your work.",
        QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  QString sourceUid = itsSelectAnnotationSourceDialog.getSourceUid();
  if(sourceUid == "-1")
  {
    QMessageBox msgBox(QMessageBox::Warning, "No source selected!",
        "Your work has not been saved!\n\n"
        "You must choose yourself as an annotation source before you can save your work.",
        QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  // Make sure we have a valid scene to save to
  if(itsSceneUid == "-1")
  {
      QMessageBox msgBox(QMessageBox::Warning, "Work not saved!",
          "Your work has not been saved!\n\n"
          "You must choose a valid scene in order to save your work.",
          QMessageBox::Ok);
      msgBox.exec();
      return;
  }

  // Clear out all entries from this annotator for this scene
  db.exec("DELETE from Polygon WHERE scene="+itsSceneUid+" AND annotationSource="+sourceUid);
  db.exec("DELETE from PolygonKeyframe WHERE scene="+itsSceneUid+" AND annotationSource="+sourceUid);

  AnnotationObjectManager* mgr = static_cast<MainWindow*>(parent())->getObjectManager();

  // Write the keyframes to the database
  std::map<int, QString> objIdMap; // A mapping from GUI ids to database ids
  QList<AnnotationObject *> objects = mgr->getAnnotationObjects(); 
  QList<AnnotationObject *>::iterator objIt; 
  for(objIt = objects.begin(); objIt != objects.end(); ++objIt)
  {
    AnnotationObject* obj                   = *objIt;
    ObjectAnimation* mainAnimation          = obj->getAnimation();
    QList<AnnotationObjectVertex*> vertices = *(obj->getVertices());


    std::map<int, PolygonKeyframe> keyframes;
    

    size_t numVertices = obj->getVertices()->size();

    QMap<int, ObjectAnimation::FrameState> keyFrames = mainAnimation->getKeyFrames();
    
    // First, insert the main polygon's keyframes
    QMap<int, ObjectAnimation::FrameState>::iterator keyIt;
    for(keyIt = mainAnimation->getKeyFrames().begin(); keyIt != mainAnimation->getKeyFrames().end(); ++keyIt)
    {
      int frameNum = keyIt.key();
      ObjectAnimation::FrameState frameState = keyIt.value();

      keyframes[frameNum].keyframe = true;
      keyframes[frameNum].visible  = frameState.visible;
      keyframes[frameNum].pos      = frameState.pos;

      keyframes[frameNum].vertices.resize(numVertices);
      keyframes[frameNum].verticesVisible.resize(numVertices);
      keyframes[frameNum].verticesKeyframe.resize(numVertices);

    }

    // Next, insert each of the vertices keyframes
    for(int vertIdx=0; vertIdx < vertices.size(); ++vertIdx)
    {
      AnnotationObjectVertex* vertex = vertices[vertIdx];
      for(keyIt=vertex->getAnimation()->getKeyFrames().begin(); keyIt!=vertex->getAnimation()->getKeyFrames().end(); ++keyIt)
      {
        int frameNum = keyIt.key();
        ObjectAnimation::FrameState frameState = keyIt.value();


        if(keyframes[frameNum].vertices.size() != numVertices)
        {
          keyframes[frameNum].vertices.resize(numVertices);
          keyframes[frameNum].verticesVisible.resize(numVertices);
          keyframes[frameNum].verticesKeyframe.resize(numVertices);
        }

        keyframes[frameNum].vertices[vertIdx]         = frameState.pos;
        keyframes[frameNum].verticesVisible[vertIdx]  = frameState.visible;
        keyframes[frameNum].verticesKeyframe[vertIdx] = true;
      }
    } // for vertex

    //Create a db entry for this object
		LINFO("####### Inserting Object (%s) as Type (%s)", obj->getDescription().toStdString().c_str(), QString::number(obj->getType()).toStdString().c_str());
    db.exec("INSERT into Object (category) VALUES ("+QString::number(obj->getType())+")");

    // Get the most recently obtained object uid - this should be safe,
    // even for multiple users
    QSqlQuery query = db.exec("select currval('object_uid_seq')");
    query.next();
    QString objDbId = query.value(0).toString();
    objIdMap[obj->getId()] = objDbId;

    //Insert the object description
    QSqlQuery descPropUidQuery = db.exec("SELECT uid FROM objectproptype WHERE name='Description'");
    descPropUidQuery.next();
    QString descPropUid = descPropUidQuery.value(0).toString();
    QSqlQuery insertDescQuery = db.exec("INSERT into objectproperties (object, type, value) VALUES ("
        +objDbId+", "
        +descPropUid+", "
        + "'" + obj->getDescription()+"')"
        );

    //Now, insert the whole keyframe structure into the database
    std::map<int, PolygonKeyframe>::iterator keyframeIt;
    for(keyframeIt=keyframes.begin();keyframeIt!=keyframes.end(); ++keyframeIt)
    {
      int frameNum = keyframeIt->first;
      PolygonKeyframe polyData = keyframeIt->second;

      QString queryString = "INSERT into PolygonKeyframe ";
      queryString += "(scene, frame, vertices, verticesVisible, verticesKeyframe, "
        "annotationSource, object, pos, visible, keyframe, time)";
      queryString += " VALUES (";
      queryString += itsSceneUid + ", ";
      queryString += QString::number(frameNum) + ", ";
      queryString += "'(";
      for(size_t i=0; i<polyData.vertices.size(); ++i)
      {
        queryString += "(" + QString::number(polyData.vertices[i].x()) + "," + QString::number(polyData.vertices[i].y()) + ")";
        if(i < polyData.vertices.size()-1) queryString += ",";
      }
      queryString += ")', ";
      queryString += "'{";
      for(size_t i=0; i<polyData.verticesVisible.size(); ++i)
      {
        queryString += polyData.verticesVisible[i] ? "t" : "f";
        if(i < polyData.vertices.size()-1) queryString += ",";
      }
      queryString += "}', ";
      queryString += "'{";
      for(size_t i=0; i<polyData.verticesKeyframe.size(); ++i)
      {
        queryString += polyData.verticesKeyframe[i] ? "t" : "f";
        if(i < polyData.vertices.size()-1) queryString += ",";
      }
      queryString += "}', ";
      queryString += sourceUid + ", ";
      queryString += objDbId + ", ";
      queryString += "'(" + QString::number(polyData.pos.x()) + "," + QString::number(polyData.pos.y()) + ")', ";
      queryString += polyData.visible ? "'t'" : "'f'";
      queryString += ", ";
      queryString += polyData.keyframe ? "'t'" : "'f'";
      queryString += ", ";
      queryString += "NOW()";
      queryString += ")";
        
      db.exec(queryString);
    }
  } // for object

  //////////////////////////////////////////////////////////////////////
  //Dump the rendered animation
  std::set<int> insertedObjects;
  std::map<int, std::map<int,AnnotationObjectFrame> > animation = mgr->renderAnimations();
  std::map<int, std::map<int,AnnotationObjectFrame> >::iterator animIt = animation.begin();
  for(;animIt!=animation.end(); ++animIt)
  {
    int frameNum = animIt->first;
    std::map<int,AnnotationObjectFrame>::iterator objectIt = animIt->second.begin();
    for(;objectIt!=animIt->second.end(); ++objectIt)
    {
      QString objId = objIdMap[objectIt->first]; //All objects already in db
      AnnotationObjectFrame objFrame = objectIt->second;
      ObjectAnimation::FrameState objFrameState = objFrame.ObjectFrameState;
      if(objFrameState.visible)
      {
        //Construct the string of vertices
        QPointF center(0, 0); int numVisVert = 0;
        QString verticesString = "(";
        std::map<int, ObjectAnimation::FrameState>::iterator vertIt = objFrame.VertexFrames.begin();
        for(; vertIt!=objFrame.VertexFrames.end(); ++vertIt)
        {
          ObjectAnimation::FrameState vertState = vertIt->second;
          if(vertState.visible)
          {
            QPointF pos = vertState.pos;//+objFrameState.pos;
            center += pos; numVisVert++;
            verticesString += "("+QString::number(pos.x())+","+QString::number(pos.y())+"),";
          }
        }
        verticesString.chop(1); //Remove last comma
        verticesString+=")";
        if(numVisVert == 0) continue;

        //Construct Center string (don't insert it for now)
        center/=numVisVert;
        QString centerString = "("+QString::number(center.x())+","+QString::number(center.y())+")";

        QString queryString = "INSERT INTO polygon (scene, frame, vertices, annotationsource, object, time) VALUES (";
        queryString += itsSceneUid + ", ";
        queryString += QString::number(frameNum) + ", ";
        queryString += "'" + verticesString + "', ";
        queryString += sourceUid + ", ";
        queryString += objId + ", "; 
        queryString += "NOW()";
        queryString += ")";
        db.exec(queryString);
      } //objFrameState.visible

    }
  }

}


void DBManager::openAnnotation()
{
  if(!connected) connectToDb();
  if(!connected)
  {
    return;
  }

  QString sourceUid = itsSelectAnnotationSourceDialog.getSourceUid();
  while(sourceUid == "-1") 
  {
    chooseAnnotationSource();
    sourceUid = itsSelectAnnotationSourceDialog.getSourceUid();
  }

  if(!itsOpenDBEntryDialog.exec()) return;
  itsSceneUid = itsOpenDBEntryDialog.getSceneUid();
  if(itsSceneUid == "-1") return;

  // Open the video file
  emit openVideo(itsOpenDBEntryDialog.getMgzJFileName());
  
  MainWindow* mw = static_cast<MainWindow*>(parent());
  AnnotationObjectManager* mgr = mw->getObjectManager();

  //Delete all existing objects
  mgr->clear();
  
  FrameRange frameRange = mw->getFrameRange();

  QSqlQuery objectQuery =
    db.exec("SELECT DISTINCT object from PolygonKeyframe WHERE scene="+itsSceneUid+" and annotationsource="+sourceUid);
  while(objectQuery.next())
  {
    QString objId = objectQuery.value(0).toString();
    QString keyframeQueryString = "SELECT frame, vertices, verticesVisible, verticesKeyFrame, pos, visible, keyframe " + 
          QString("from PolygonKeyframe WHERE scene=")+itsSceneUid+
          " and object="+objId+
          " order by frame";
    QSqlQuery keyframeQuery = db.exec(keyframeQueryString);
    if(keyframeQuery.size() == 0) continue;

    AnnotationObject* obj = NULL; // new AnnotationObject(keyframeQuery.value(0).toString(), frameRange, );

    while(keyframeQuery.next())
    {
      //Get the frame number
      int frame = keyframeQuery.value(0).toInt();

      //Get the list of vertices
      std::vector<QPointF> vertices;
      QString verticesString = keyframeQuery.value(1).toString().mid(1, keyframeQuery.value(1).toString().size()-2);
      int stringpos = verticesString.indexOf("(")+1;
      int stringendpos;
      while((stringendpos = verticesString.indexOf(")", stringpos)) != -1)
      {
        QString vertexString = verticesString.mid(stringpos, stringendpos-stringpos);
        stringpos = verticesString.indexOf("(",stringendpos)+1;

        QStringList coordStrings = vertexString.split(",");
        QPointF vPos(coordStrings[0].toFloat(), coordStrings[1].toFloat());
        vertices.push_back(vPos);

        if(stringpos == 0) break;
      }

      //Get the vertices visible list
      QStringList verticesVisible =
        keyframeQuery.value(2).toString().mid(1, keyframeQuery.value(2).toString().size()-2).split(",");

      //Get the vertices keyframe list
      QStringList verticesKeyframe =
        keyframeQuery.value(3).toString().mid(1, keyframeQuery.value(3).toString().size()-2).split(",");

      //Get the position
      QStringList posStringCoords = 
        keyframeQuery.value(4).toString().mid(1, keyframeQuery.value(4).toString().size()-2).split(",");
      QPointF pos(posStringCoords[0].toFloat(), posStringCoords[1].toFloat());

      //Get the visibility
      QString visible = keyframeQuery.value(5).toString();

      //Get the visibility
      QString keyframe = keyframeQuery.value(6).toString();

      //Create the new object if it is not yet made
      if(obj == NULL)
      {
        //Find the object's category
        QSqlQuery objInfoQuery = db.exec("SELECT category FROM object WHERE uid="+objId);
        objInfoQuery.next();
        int category = objInfoQuery.value(0).toInt();

        QSqlQuery descPropUidQuery = db.exec("SELECT uid FROM objectproptype WHERE name='Description'");
        descPropUidQuery.next();
        QString descPropUid = descPropUidQuery.value(0).toString();
        QSqlQuery descQuery = db.exec("SELECT value FROM objectproperties WHERE object="+objId+" and type="+descPropUid);
        descQuery.next();
        QString description = descQuery.value(0).toString();


        //TODO: FIXME object name
        obj = new AnnotationObject(frame, frameRange, pos, description, category);

        //Delete the auto-created vertices
        for(int vIdx=0; vIdx < obj->getVertices()->size(); ++vIdx)
          delete obj->getVertices()->at(vIdx);
        obj->getVertices()->clear();

        //Add in the vertices
        for(size_t vIdx=0; vIdx<vertices.size(); ++vIdx)
        {
          AnnotationObjectVertex* vert = new AnnotationObjectVertex(obj, frame, frameRange, vertices[vIdx]);
          //if(verticesVisible[vIdx] == "f") vert->setKeyframe(frame, vertices[vIdx], false);
          obj->addVertex(vIdx, vert);
        }
      }

      if(keyframe[0] == 't')
        obj->setKeyframe(frame, pos, (visible[0] == 't'));

      for(size_t vIdx=0; vIdx<vertices.size(); ++vIdx)
        if(verticesKeyframe[vIdx][0] == 't')
          obj->getVertexById(vIdx)->setKeyframe(frame, vertices[vIdx], (verticesVisible[vIdx][0] == 't'));
    }

    //Add the object to our manager
    mgr->addObject(obj);
    mw->getMainDisplay()->addObject(obj);
    connect(mw->getTimeline(), SIGNAL(frameChanged(int)), obj, SLOT(frameChanged(int)));
  }
  mw->getTimeline()->setCurrentTime(0);
}

QMap<int, QString> DBManager::getObjCategories()
{

  QSqlQuery catQuery =
    db.exec("SELECT uid, name FROM objectcategory where type <> 4");

  QMap<int, QString> categories;
  while(catQuery.next())
    categories[catQuery.value(0).toInt()] = catQuery.value(1).toString();

  return categories;
}

void DBManager::updateSettings(QSettings *settings)
{
  qDebug() << "Loading Settings:";
  if(settings->contains("archiveLoc"))
  {
    itsArchiveLoc  = settings->value("archiveLoc").toString();
    qDebug() << "   itsArchiveLoc  " << itsArchiveLoc; 
  }
  if(settings->contains("workingLoc"))
  {
    itsWorkingLoc  = settings->value("workingLoc").toString();
    qDebug() << "   itsWorkingLoc  " << itsWorkingLoc;  
  }
  if(settings->contains("incomingLoc"))
  {
    itsIncomingLoc = settings->value("incomingLoc").toString();
    qDebug() << "   itsIncomingLoc " << itsIncomingLoc; 
  }
}
