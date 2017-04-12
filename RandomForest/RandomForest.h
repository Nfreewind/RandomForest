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
		std::vector<unsigned char> data;
		unsigned char label;
	};

	class DecisionTreeNode {
	public:
		int split_attribute_id;
		unsigned char label;
		int depth;
		QMap<int, boost::shared_ptr<DecisionTreeNode>> children;

	public:
		DecisionTreeNode(int depth);

		unsigned char test(const boost::shared_ptr<Example>& example);
		QDomElement save(QDomDocument& doc);
		unsigned char setLabelFromChildren();
	};

	class DecisionTree {
	private:
		boost::shared_ptr<DecisionTreeNode> root;

	public:
		DecisionTree();

		void construct(const std::vector<boost::shared_ptr<Example>>& examples, bool sample_attributes, int max_depth);
		int test(const boost::shared_ptr<Example>& example);
		void save(const QString& filename);
		QDomElement save(QDomDocument& doc);

	private:
		boost::shared_ptr<DecisionTreeNode> constructNodes(const std::vector<boost::shared_ptr<Example>>& examples, int depth, bool sample_attributes, int max_depth);
		float calculateEntropy(const std::vector<boost::shared_ptr<Example>>& examples, int split_attribute);
	};

	class RandomForest {
	private:
		std::vector<DecisionTree> trees;

	public:
		RandomForest();

		void construct(const std::vector<boost::shared_ptr<Example>>& examples, int num_trees, float ratio, int max_depth);
		void save(const QString& filename);
		int test(const boost::shared_ptr<Example>& example);
	};

}