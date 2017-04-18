#include "MainWindow.h"
#include <QDir>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "RandomForest.h"
#include <time.h>

cv::Vec3b convertLabelToColor(unsigned char label) {
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

unsigned char convertColorToLabel(const cv::Vec3b& color) {
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

boost::shared_ptr<rf::Example> extractExampleFromPatch(const cv::Mat& patch, const cv::Vec3b& ground_truth) {
	boost::shared_ptr<rf::Example> example = boost::shared_ptr<rf::Example>(new rf::Example());

	for (int index = 0; index < patch.rows * patch.cols; ++index) {
		int y = index / patch.cols;
		int x = index % patch.cols;

		cv::Vec3b col = patch.at<cv::Vec3b>(y, x);

		float val = ((float)col[0] + (float)col[1] + (float)col[2]) / 3.0f / 25.6;
		if (val >= 10) val = 9;
		if (val < 0) val = 0;

		example->data.push_back(val);
	}

	example->label = convertColorToLabel(ground_truth);

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

	time_t start = clock();
	QDir ground_truth_dir("../ECP/ground_truth/");
	QDir train_images_dir("../ECP/images_train/");

	printf("Image processing for training dataset: ");
	QStringList train_image_files = train_images_dir.entryList(QDir::NoDotAndDotDot | QDir::Files);// , QDir::DirsFirst);
	std::vector<boost::shared_ptr<rf::Example>> examples;
	for (int i = 0; i < train_image_files.size(); ++i) {
		printf("\rImage processing for training dataset: %d", i + 1);

		// remove the file extension
		int index = train_image_files[i].lastIndexOf(".");
		QString filename = train_image_files[i].left(index);

		//std::cout << image_file.toUtf8().constData() << std::endl;
		cv::Mat image = cv::imread((train_images_dir.absolutePath() + "/" + filename + ".jpg").toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData());
		//std::cout << "(" << image.rows << " x " << image.cols << ")" << std::endl;

		for (int y = 0; y < image.rows - patch_size + 1; y++) {
			for (int x = 0; x < image.cols - patch_size + 1; x++) {
				cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
				//std::cout << roi.rows << "," << roi.cols << std::endl;
				cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);
				
				boost::shared_ptr<rf::Example> example = extractExampleFromPatch(image_roi, ground_truth_color);
				examples.push_back(example);
			}
		}
	}
	printf("\n");

	time_t end = clock();

	std::cout << "Dataset has been created." << std::endl;
	std::cout << "#examples: " << examples.size() << std::endl;
	std::cout << "#attributes: " << examples[0]->data.size() << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;

	// create random forest
	start = clock();
	QMap<unsigned char, float> priors;
	priors[rf::Example::LABEL_WALL] = 1;
	priors[rf::Example::LABEL_WINDOW] = 1.8;
	priors[rf::Example::LABEL_DOOR] = 4;
	priors[rf::Example::LABEL_BALCONY] = 2;
	priors[rf::Example::LABEL_SHOP] = 1.5;
	priors[rf::Example::LABEL_ROOF] = 3.4;
	priors[rf::Example::LABEL_SKY] = 1.5;
	priors[rf::Example::LABEL_UNKNOWN] = 0;
	rf::RandomForest rand_forest;
	rand_forest.construct(examples, T, r, max_depth, priors);
	//rand_forest.save("forest.xml");
	end = clock();
	std::cout << "Random forest has been created." << std::endl;
	std::cout << "Elapsed: " << (end - start) / CLOCKS_PER_SEC << " sec." << std::endl;


	// release the memory for the training data
	examples.clear();


	// test
	QDir test_images_dir("../ECP/images_test/");

	start = clock();
	QDir result_dir("results/");
	cv::Mat confusionMatrix(7, 7, CV_32F, cv::Scalar(0.0f));
	printf("Testing: ");
	QStringList test_image_files = test_images_dir.entryList(QDir::NoDotAndDotDot | QDir::Files);// , QDir::DirsFirst);
	for (int i = 0; i < test_image_files.size(); ++i) {
		printf("\rTesting: %d", i + 1);

		// remove the file extension
		int index = test_image_files[i].lastIndexOf(".");
		QString filename = test_image_files[i].left(index);

		cv::Mat image = cv::imread((test_images_dir.absolutePath() + "/" + filename + ".jpg").toUtf8().constData());
		cv::Mat ground_truth = cv::imread((ground_truth_dir.absolutePath() + "/" + filename + ".png").toUtf8().constData());

		cv::Mat result(image.size(), image.type(), cv::Vec3b(0, 0, 0));
		for (int y = 0; y < image.rows - patch_size + 1; y++) {
			for (int x = 0; x < image.cols - patch_size + 1; x++) {
				cv::Mat image_roi = image(cv::Rect(x, y, patch_size, patch_size));
				boost::shared_ptr<rf::Example> example = extractExampleFromPatch(image_roi, cv::Vec3b(0, 0, 0));
				unsigned char label = rand_forest.test(example);

				// HACK
				// if the label cannot be estimated, assume it is wall
				if (label == rf::Example::LABEL_UNKNOWN) {
					label = rf::Example::LABEL_WALL;
				}
				result.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2) = convertLabelToColor(label);


				cv::Vec3b ground_truth_color = ground_truth.at<cv::Vec3b>(y + (patch_size - 1) / 2, x + (patch_size - 1) / 2);
				unsigned char ground_truth_label = convertColorToLabel(ground_truth_color);

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
	std::vector<boost::shared_ptr<rf::Example>> examples;
	
	boost::shared_ptr<rf::Example> example = boost::shared_ptr<rf::Example>(new rf::Example());
	example->label = 2;
	example->data.push_back(0);
	example->data.push_back(0);
	example->data.push_back(0);
	example->data.push_back(0);
	example->data.push_back(0);
	example->data.push_back(2);
	examples.push_back(example);

	boost::shared_ptr<rf::Example> example2 = boost::shared_ptr<rf::Example>(new rf::Example());
	example2->label = 1;
	example2->data.push_back(2);
	example2->data.push_back(0);
	example2->data.push_back(0);
	example2->data.push_back(0);
	example2->data.push_back(0);
	example2->data.push_back(1);
	examples.push_back(example2);

	boost::shared_ptr<rf::Example> example3 = boost::shared_ptr<rf::Example>(new rf::Example());
	example3->label = 1;
	example3->data.push_back(1);
	example3->data.push_back(2);
	example3->data.push_back(0);
	example3->data.push_back(0);
	example3->data.push_back(0);
	example3->data.push_back(1);
	examples.push_back(example3);

	boost::shared_ptr<rf::Example> example4 = boost::shared_ptr<rf::Example>(new rf::Example());
	example4->label = 1;
	example4->data.push_back(1);
	example4->data.push_back(1);
	example4->data.push_back(0);
	example4->data.push_back(0);
	example4->data.push_back(0);
	example4->data.push_back(1);
	examples.push_back(example4);

	boost::shared_ptr<rf::Example> example5 = boost::shared_ptr<rf::Example>(new rf::Example());
	example5->label = 1;
	example5->data.push_back(1);
	example5->data.push_back(3);
	example5->data.push_back(2);
	example5->data.push_back(2);
	example5->data.push_back(0);
	example5->data.push_back(1);
	examples.push_back(example5);

	boost::shared_ptr<rf::Example> example6 = boost::shared_ptr<rf::Example>(new rf::Example());
	example6->label = 1;
	example6->data.push_back(0);
	example6->data.push_back(0);
	example6->data.push_back(0);
	example6->data.push_back(0);
	example6->data.push_back(4);
	example6->data.push_back(1);
	examples.push_back(example6);

	boost::shared_ptr<rf::Example> example7 = boost::shared_ptr<rf::Example>(new rf::Example());
	example7->label = 2;
	example7->data.push_back(1);
	example7->data.push_back(4);
	example7->data.push_back(0);
	example7->data.push_back(0);
	example7->data.push_back(1);
	example7->data.push_back(1);
	examples.push_back(example7);

	boost::shared_ptr<rf::Example> example8 = boost::shared_ptr<rf::Example>(new rf::Example());
	example8->label = 2;
	example8->data.push_back(1);
	example8->data.push_back(4);
	example8->data.push_back(0);
	example8->data.push_back(0);
	example8->data.push_back(2);
	example8->data.push_back(1);
	examples.push_back(example8);

	boost::shared_ptr<rf::Example> example9 = boost::shared_ptr<rf::Example>(new rf::Example());
	example9->label = 2;
	example9->data.push_back(1);
	example9->data.push_back(4);
	example9->data.push_back(0);
	example9->data.push_back(0);
	example9->data.push_back(3);
	example9->data.push_back(1);
	examples.push_back(example9);
	
	boost::shared_ptr<rf::Example> example10 = boost::shared_ptr<rf::Example>(new rf::Example());
	example10->label = 2;
	example10->data.push_back(1);
	example10->data.push_back(3);
	example10->data.push_back(1);
	example10->data.push_back(1);
	example10->data.push_back(1);
	example10->data.push_back(1);
	examples.push_back(example10);

	boost::shared_ptr<rf::Example> example11 = boost::shared_ptr<rf::Example>(new rf::Example());
	example11->label = 2;
	example11->data.push_back(1);
	example11->data.push_back(3);
	example11->data.push_back(1);
	example11->data.push_back(1);
	example11->data.push_back(2);
	example11->data.push_back(1);
	examples.push_back(example11);

	boost::shared_ptr<rf::Example> example12 = boost::shared_ptr<rf::Example>(new rf::Example());
	example12->label = 2;
	example12->data.push_back(1);
	example12->data.push_back(3);
	example12->data.push_back(1);
	example12->data.push_back(2);
	example12->data.push_back(1);
	example12->data.push_back(1);
	examples.push_back(example12);

	boost::shared_ptr<rf::Example> example13 = boost::shared_ptr<rf::Example>(new rf::Example());
	example13->label = 2;
	example13->data.push_back(1);
	example13->data.push_back(3);
	example13->data.push_back(1);
	example13->data.push_back(2);
	example13->data.push_back(2);
	example13->data.push_back(1);
	examples.push_back(example13);

	boost::shared_ptr<rf::Example> example14 = boost::shared_ptr<rf::Example>(new rf::Example());
	example14->label = 1;
	example14->data.push_back(1);
	example14->data.push_back(3);
	example14->data.push_back(1);
	example14->data.push_back(1);
	example14->data.push_back(3);
	example14->data.push_back(1);
	examples.push_back(example14);

	boost::shared_ptr<rf::Example> example15 = boost::shared_ptr<rf::Example>(new rf::Example());
	example15->label = 2;
	example15->data.push_back(1);
	example15->data.push_back(3);
	example15->data.push_back(1);
	example15->data.push_back(2);
	example15->data.push_back(3);
	example15->data.push_back(1);
	examples.push_back(example15);

	rf::DecisionTree dt;
	dt.construct(examples, false, 2, QMap<unsigned char, float>());
	dt.save("test.xml");

	boost::shared_ptr<rf::Example> example16 = boost::shared_ptr<rf::Example>(new rf::Example());
	example16->data.push_back(2);
	example16->data.push_back(3);
	example16->data.push_back(2);
	example16->data.push_back(1);
	example16->data.push_back(2);
	example16->data.push_back(1);
	std::cout << dt.test(example16) << std::endl;

	boost::shared_ptr<rf::Example> example17 = boost::shared_ptr<rf::Example>(new rf::Example());
	example17->data.push_back(2);
	example17->data.push_back(3);
	example17->data.push_back(2);
	example17->data.push_back(1);
	example17->data.push_back(3);
	example17->data.push_back(1);
	std::cout << dt.test(example17) << std::endl;

	boost::shared_ptr<rf::Example> example18 = boost::shared_ptr<rf::Example>(new rf::Example());
	example18->data.push_back(0);
	example18->data.push_back(3);
	example18->data.push_back(2);
	example18->data.push_back(1);
	example18->data.push_back(0);
	example18->data.push_back(1);
	std::cout << dt.test(example18) << std::endl;
}