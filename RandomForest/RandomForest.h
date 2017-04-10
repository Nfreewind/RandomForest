#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>
#include <QMap>
#include <QString>
#include <QDomElement>

namespace rf {
	class Example {
	public:
		static enum { LABEL_WALL = 0, LABEL_WINDOW, LABEL_DOOR, LABEL_BALCONY, LABEL_SHOP, LABEL_ROOF, LABEL_SKY, LABEL_UNKNOWN };

	public:
		std::vector<int> data;
		int label;
	};

	class DecisionTreeNode {
	public:
		int split_attribute_id;
		int label;
		int depth;
		QMap<int, boost::shared_ptr<DecisionTreeNode>> children;

	public:
		DecisionTreeNode(int depth);

		int test(const Example& example);
		QDomElement save(QDomDocument& doc);
	};

	class DecisionTree {
	private:
		boost::shared_ptr<DecisionTreeNode> root;

	public:
		DecisionTree();

		void construct(const std::vector<Example>& examples, bool sample_attributes, int max_depth);
		int test(const Example& example);
		void save(const QString& filename);
		QDomElement save(QDomDocument& doc);

	private:
		boost::shared_ptr<DecisionTreeNode> constructNodes(const std::vector<Example>& examples, int depth, bool sample_attributes, int max_depth);
		float calculateEntropy(const std::vector<Example>& examples, int split_attribute);
	};

	class RandomForest {
	private:
		std::vector<DecisionTree> trees;

	public:
		RandomForest();

		void construct(const std::vector<Example>& examples, int num_trees, float ratio, int max_depth);
		void save(const QString& filename);
		int test(const Example& example);
	};

}