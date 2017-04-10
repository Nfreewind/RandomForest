#include "MainWindow.h"
#include <QDir>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "RandomForest.h"
#include <time.h>

cv::Vec3b convertLabelToColor(int label) {
	if (label == rf::Example::LABEL_WALL) {
		return cv::Vec3b(0, 255, 255);
	}
	else if (label == rf::Example::LABEL_WINDOW) {
		return cv::Vec3b(0, 0, 255);
	}
	else if (label == rf::Example::LABEL_DOOR) {
		return cv::Vec3b(0, 128, 255);
	}
	else if (label == rf::Example::LABEL_BALCONY) {
		return cv::Vec3b(255, 0, 128);
	}
	else if (label == rf::Example::LABEL_SHOP) {
		return cv::Vec3b(0, 255, 0);
	}
	else if (label == rf::Example::LABEL_ROOF) {
		return cv::Vec3b(255, 0, 0);
	}
	else if (label == rf::Example::LABEL_SKY) {
		return cv::Vec3b(255, 255, 128);
	}
	else {
		//return cv::Vec3b(0, 0, 0);
		// HACK
		// if the label is unknown, assume it is wall.
		return cv::Vec3b(0, 255, 255);
	}
}

int convertColorToLabel(const cv::Vec3b& color) {
	if (color == cv::Vec3b(0, 255, 255)) {
		return rf::Example::LABEL_WALL;
	}
	else if (color == cv::Vec3b(0, 0, 255)) {
		return rf::Example::LABEL_WINDOW;
	}
	else if (color == cv::Vec3b(0, 128, 255)) {
		return rf::Example::LABEL_DOOR;
	}
	else if (color == cv::Vec3b(255, 0, 128)) {
		return rf::Example::LABEL_BALCONY;
	}
	else if (color == cv::Vec3b(0, 255, 0)) {
		return rf::Example::LABEL_SHOP;
	}
	else if (color == cv::Vec3b(255, 0, 0)) {
		return rf::Example::LABEL_ROOF;
	}
	else if (color == cv::Vec3b(255, 255, 128)) {
		return rf::Example::LABEL_SKY;
	}
	else {
		return rf::Example::LABEL_UNKNOWN;
	}
}

rf::Example extractExampleFromPatch(const cv::Mat& patch, const cv::Vec3b& ground_truth) {
	rf::Example example;

	for (int index = 0; index < patch.rows * patch.cols; ++index) {
		int y = index / patch.cols;
		int x = index % patch.cols;

		cv::Vec3b col = patch.at<cv::Vec3b>(y, x);

		float val = ((float)col[0] + (float)col[1] + (float)col[2]) / 3.0f / 25.6;
		if (val >= 10) val = 9;
		if (val < 0) val = 0;

		example.data.push_back(val);
	}

	example.label = convertColorToLabel(ground_truth);

	return example;
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	ui.setupUi(this);

	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));
	connect(ui.actionTrainByECP, SIGNAL(triggered()), this, SLOT(onTrainByECP()));
	connect(ui.actionDecisionTreeTest, SIGNAL(triggered()), this, SLOT(onDecisionTreeTest()));
}

MainWindow::~MainWindow() {
}

void MainWindow::onTrainByECP() {
	const int patch_size = 15;
	const int T = 10;
	const float r = 0.5;
	const int max_depth = 18;

	// create dataset
	time_t start = clock();
	QDir ground_truth_dir("../ECP/ground_truth/");
	QDir images_dir("../ECP/images/");

	std::vector<rf::Example> examples;

	QStringList image_files = images_dir.entryList(QDir::NoDotAndDotDot | QDir::Files);// , QDir::DirsFirst);
	printf("Image processing: ");
	for (int i = 0; i < image_files.size(); ++i) {
		printf("\rImage processing: %d", i + 1);

		// remove the file extension
		int index = image_files[i].lastIndexOf(".");
		QString filename = image_files[i].left(index);

		//std::cout << image_file.toUtf8().constData() << std::endl;
		cv::Mat image = cv::imread((images_dir.absolutePath() + "/" + image_files[i]).toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData());
		//std::cout << "(" << image.rows << " x " << image.cols << ")" << std::endl;

		for (int y = 0; y < image.rows - patch_size + 1; y++) {
			for (int x = 0; x < image.cols - patch_size + 1; x++) {
				cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
				//std::cout << roi.rows << "," << roi.cols << std::endl;
				cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);
				
				rf::Example example = extractExampleFromPatch(image_roi, ground_truth_color);
				examples.push_back(example);
			}
		}
	}
	printf("\n");

	time_t end = clock();

	std::cout << "Dataset has been created." << std::endl;
	std::cout << "#examples: " << examples.size() << std::endl;
	std::cout << "#attributes: " << examples[0].data.size() << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;

	// create random forest
	start = clock();
	rf::RandomForest rand_forest;
	rand_forest.construct(examples, T, r, max_depth);
	//rand_forest.save("forest.xml");
	end = clock();
	std::cout << "Random forest has been created." << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;

	// test
	start = clock();
	QDir result_dir("results/");
	cv::Mat confusionMatrix(7, 7, CV_32F, cv::Scalar(0.0f));
	printf("Testing: ");
	for (int i = 0; i < image_files.size(); ++i) {
		printf("\rTesting: %d", i + 1);

		// remove the file extension
		int index = image_files[i].lastIndexOf(".");
		QString filename = image_files[i].left(index);

		cv::Mat image = cv::imread((images_dir.absolutePath() + "/" + filename + ".jpg").toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData());

		cv::Mat result(image.size(), image.type(), cv::Vec3b(0, 0, 0));
		for (int y = 0; y < image.rows - patch_size + 1; y++) {
			for (int x = 0; x < image.cols - patch_size + 1; x++) {
				cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
				cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);
				int ground_truth_label = convertColorToLabel(ground_truth_color);

				rf::Example example = extractExampleFromPatch(image_roi, cv::Vec3b(0, 0, 0));
				
				int label = rand_forest.test(example);
				// HACK
				// if the label cannot be estimated, assume it is wall
				if (label < 0 || label > rf::Example::LABEL_UNKNOWN) {
					label = rf::Example::LABEL_WALL;
				}
				result.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2) = convertLabelToColor(label);

				// update confusion matrix
				confusionMatrix.at<float>(ground_truth_label, label) += 1;
			}
		}

		cv::imwrite((result_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData(), result);
	}
	printf("\n");
	end = clock();
	std::cout << "Test has been finished." << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;

	std::cout << "Confusion matrix:" << std::endl;
	cv::Mat confusionMatrixSum;
	cv::reduce(confusionMatrix, confusionMatrixSum, 1, cv::REDUCE_SUM);
	for (int r = 0; r < confusionMatrix.rows; ++r) {
		for (int c = 0; c < confusionMatrix.cols; ++c) {
			if (c > 0) std::cout << ", ";
			std::cout << confusionMatrix.at<float>(r, c) / confusionMatrixSum.at<float>(r, 0);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void MainWindow::onDecisionTreeTest() {
	std::vector<rf::Example> examples;
	
	rf::Example example;
	example.label = 2;
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(2);
	examples.push_back(example);

	example.label = 1;
	example.data.clear();
	example.data.push_back(2);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 1;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(2);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 1;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(1);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 1;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(2);
	example.data.push_back(2);
	example.data.push_back(0);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 1;
	example.data.clear();
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(4);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(4);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(1);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(4);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(2);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(4);
	example.data.push_back(0);
	example.data.push_back(0);
	example.data.push_back(3);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	example.data.push_back(1);
	example.data.push_back(1);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	example.data.push_back(1);
	example.data.push_back(2);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	example.data.push_back(2);
	example.data.push_back(1);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	example.data.push_back(2);
	example.data.push_back(2);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 1;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	examples.push_back(example);

	example.label = 2;
	example.data.clear();
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	example.data.push_back(2);
	example.data.push_back(3);
	example.data.push_back(1);
	examples.push_back(example);

	rf::DecisionTree dt;
	dt.construct(examples, false, 2);
	dt.save("test.xml");

	example.data.clear();
	example.data.push_back(2);
	example.data.push_back(3);
	example.data.push_back(2);
	example.data.push_back(1);
	example.data.push_back(2);
	example.data.push_back(1);
	std::cout << dt.test(example) << std::endl;

	example.data.clear();
	example.data.push_back(2);
	example.data.push_back(3);
	example.data.push_back(2);
	example.data.push_back(1);
	example.data.push_back(3);
	example.data.push_back(1);
	std::cout << dt.test(example) << std::endl;

	example.data.clear();
	example.data.push_back(0);
	example.data.push_back(3);
	example.data.push_back(2);
	example.data.push_back(1);
	example.data.push_back(0);
	example.data.push_back(1);
	std::cout << dt.test(example) << std::endl;
}