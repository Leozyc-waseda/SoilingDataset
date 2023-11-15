#include <Qt/QtCore>
#include <Qt/QtGui>

class PreferencesDialog : public QDialog
{
  Q_OBJECT

  public:
    PreferencesDialog(QWidget* parent);

    QLineEdit* archiveLocEdit; 
    QLineEdit* workingLocEdit; 
    QLineEdit* incomingLocEdit; 
};
