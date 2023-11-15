/****************************************************************************
** Form interface generated from reading ui file 'Qt/ModelManagerControl.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef MODELMANAGERCONTROL_H
#define MODELMANAGERCONTROL_H

#include <qvariant.h>
#include <qdialog.h>
#include "Component/ModelManager.H"
#include "ModelManagerDialog.h"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;

class ModelManagerControl : public QDialog
{
    Q_OBJECT

public:
    ModelManagerControl( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~ModelManagerControl();

    QPushButton* saveButton;
    QPushButton* loadButton;
    QPushButton* startstopButton;
    QPushButton* exitButton;
    QPushButton* configButton;

public slots:
    virtual void showConfigDialog( void );
    virtual void loadConfig( void );
    virtual void saveConfig( void );
    virtual void start_or_stop( void );
    virtual void init( ModelManager & manager, bool * dorun_ );
    virtual void exitPressed( void );

protected:

protected slots:
    virtual void languageChange();

private:
    ModelManager *mgr;
    ModelManagerDialog mmd;
    bool *dorun;

};

#endif // MODELMANAGERCONTROL_H
