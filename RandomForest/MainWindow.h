#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"

class MainWindow : public QMainWindow {
	Q_OBJECT

private:
	Ui::MainWindowClass ui;

public:
	MainWindow(QWidget *parent = 0);
	~MainWindow();


public slots:
	void onTrainByECP();
	void onDecisionTreeTest();
};

#endif // MAINWINDOW_H
