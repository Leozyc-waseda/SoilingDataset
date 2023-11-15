/****************************************************************************
** Form interface generated from reading ui file 'Qt/BayesNetworkDialog.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef BAYESNETWORKDIALOG_H
#define BAYESNETWORKDIALOG_H

#include <qvariant.h>
#include <qdialog.h>
#include "Learn/Bayes.H"
#include "Qt/ExtraIncludes.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QTabWidget;
class QWidget;
class QPushButton;

class BayesNetworkDialog : public QDialog
{
    Q_OBJECT

public:
    BayesNetworkDialog( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~BayesNetworkDialog();

    QTabWidget* tabWidget;
    QWidget* tab;
    QPushButton* buttonOk;
    QPushButton* buttonCancel;

    virtual void init( Bayes & bayesNet );
    virtual void update();
    virtual void setupTab();

protected:
    Bayes *itsBayesNet;

    QVBoxLayout* BayesNetworkDialogLayout;
    QHBoxLayout* layout2;
    QSpacerItem* spacer2;

protected slots:
    virtual void languageChange();

};

#endif // BAYESNETWORKDIALOG_H
