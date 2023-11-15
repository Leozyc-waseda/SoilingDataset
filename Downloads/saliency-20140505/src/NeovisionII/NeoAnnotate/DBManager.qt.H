#ifndef NEOVISIONII_NEOANNOTATE_DBMANAGER_QT_H
#define NEOVISIONII_NEOANNOTATE_DBMANAGER_QT_H

#include <Qt/QtCore>

#include <Qt/qdialog.h>
#include <Qt/qlineedit.h>
#include <Qt/qcombobox.h>
#include <Qt/qformlayout.h>
#include <Qt/qpushbutton.h>
#include <Qt/qmessagebox.h>
#include <Qt/qdatetimeedit.h>
#include <Qt/qtreewidget.h>

#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QtSql/QSqlQuery>
#include <Qt3Support/q3networkprotocol.h>

#define VIDEO_ARCHIVE_LOCATION "/lab/dbvideos"
#define VIDEO_WORKING_LOCATION "/lab/dbvideos"
#define VIDEO_INCOMING_LOCATION "/lab/dbvideos/incoming/"

class DBManager;

class ConnectionDialog : public QDialog
{
  Q_OBJECT

  public:
    ConnectionDialog(QWidget* parent);

    QLineEdit* serverNameEdit; 
    QLineEdit* dbNameEdit; 
    QLineEdit* userNameEdit; 
    QLineEdit* passwordEdit; 
};

class NewDBEntryDialog : public QDialog
{
  Q_OBJECT 

  public:
    NewDBEntryDialog(QWidget* parent, DBManager* mgr, QSqlDatabase* _db);

    void showEvent(QShowEvent* event);

    QString getMgzJFileName() { return itsMgzJFileName; }
    QString getSceneUid()     { return itsSceneUid; }

  public slots:
    void fillInCameraOperator(int labComboBoxRow);
    void commitNewScene();
    void browse();
    void copyFinished(int exitCode, QProcess::ExitStatus exitStatus);

  signals:
    void closeProgressDialog();


  private:
    DBManager *itsDBManager;

    QLabel* filenameLabel;
    QComboBox* labComboBox;
    QComboBox* cameraComboBox;
    QComboBox* operatorComboBox;
    QComboBox* weatherComboBox;
    QLineEdit* startFrameEdit;
    QLineEdit* numFramesEdit;
    QLineEdit* frameRateEdit;
    QLineEdit* sceneNameEdit;
    QLabel* fileNameLabel;
    QDateTimeEdit* startDateTimeEdit;
    QDateTimeEdit* endDateTimeEdit;
    QComboBox* timeZoneBox;

    QString itsMgzJFileName;

    QSqlDatabase* db;

    QString itsSceneUid;

};

class OpenDBEntryDialog : public QDialog
{
  Q_OBJECT

  public:
    OpenDBEntryDialog(QWidget* parent, DBManager *mgr, QSqlDatabase* _db);

    QString getMgzJFileName() { return itsMgzJFileName; }
    QString getSceneUid()     { return itsSceneUid; }

    void showEvent(QShowEvent* event);


  public slots:
    void openEntry();

  private:
    DBManager *itsDBManager;
    QTreeWidget* itsSceneTree;
    QString itsMgzJFileName;
    QString itsSceneUid;
    QSqlDatabase* db;
};

class SelectAnnotationSourceDialog: public QDialog
{
  Q_OBJECT
  public:
    SelectAnnotationSourceDialog(QWidget* parent, QSqlDatabase* _db);

    QString getSourceUid() { return itsSourceUid; }

    void showEvent(QShowEvent* event);

  public slots:
    void selectSource();

  private:
  QTreeWidget* itsSourceTree;
  QString itsSourceUid;
  QSqlDatabase* db;
};

class DBManager : public QWidget
{
  Q_OBJECT;
  public:
    DBManager(QWidget* parent) :
      QWidget(parent),
      itsSelectAnnotationSourceDialog(parent, &db),
      itsNewDBEntryDialog(parent, this, &db),
      itsOpenDBEntryDialog(parent, this, &db),
      itsConnDialog(parent),
      connected(false),
      itsSceneUid("-1")
    { 
      itsArchiveLoc  = VIDEO_ARCHIVE_LOCATION;
      itsWorkingLoc  = VIDEO_WORKING_LOCATION;
      itsIncomingLoc = VIDEO_INCOMING_LOCATION;
    }

    void saveAnnotation();

    void createDBEntry()
    {
      if(!connected) connectToDb();
      if(!connected) return;

      if(itsNewDBEntryDialog.exec())
      {
        itsSceneUid = itsNewDBEntryDialog.getSceneUid();
        emit openVideo(itsNewDBEntryDialog.getMgzJFileName());
      }
    }


    void openAnnotation();

    bool isConnected() { return connected; }

    QMap<int, QString> getObjCategories();


    void updateSettings(QSettings *settings);
    QString itsArchiveLoc;
    QString itsWorkingLoc;
    QString itsIncomingLoc;
  signals:
    void openVideo(QString fileName);

  public slots:
    void connectToDb();
    void chooseAnnotationSource();


  private:
    QSqlDatabase db;
    SelectAnnotationSourceDialog itsSelectAnnotationSourceDialog;
    NewDBEntryDialog itsNewDBEntryDialog;
    OpenDBEntryDialog itsOpenDBEntryDialog;
    ConnectionDialog itsConnDialog;
    bool connected;
    QString itsSceneUid;

};

#endif // NEOVISIONII_NEOANNOTATE_DBMANAGER_QT_H

