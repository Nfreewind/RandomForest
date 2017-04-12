#include "RandomForest.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <QFile>
#include <QTextStream>
#include <iostream>

namespace rf {

	DecisionTreeNode::DecisionTreeNode(int depth) {
		this->depth = depth;
		split_attribute_id = -1;
		label = Example::LABEL_UNKNOWN;
	}

	unsigned char DecisionTreeNode::test(const boost::shared_ptr<Example>& example) {
		// if this is a leaf node, return the label
		if (children.size() == 0) return label;

		if (children.contains(example->data[split_attribute_id])) {
			return children[example->data[split_attribute_id]]->test(example);
		}
		else {
			// If the value does not exist in the children,
			// we use maximum vote to guess the label.
			return label;
		}
	}

	QDomElement DecisionTreeNode::save(QDomDocument& doc) {
		QDomElement node = doc.createElement("node");

		if (children.size() == 0) {
			node.setAttribute("label", label);
		}
		else {
			for (auto it = children.begin(); it != children.end(); ++it) {
				QDomElement child_node = it.value()->save(doc);
				child_node.setAttribute("value", it.key());
				node.appendChild(child_node);
			}
		}

		return node;
	}

	unsigned char DecisionTreeNode::setLabelFromChildren() {
		if (children.size() > 0) {
			QMap<unsigned char, int> votes;
			for (auto it = children.begin(); it != children.end(); ++it) {
				unsigned char label = it.value()->setLabelFromChildren();
				if (!votes.contains(label)) {
					votes[label] = 0;
				}
				votes[label]++;
			}

			int max_votes = 0;
			unsigned char max_voted_label;
			for (auto it = votes.begin(); it != votes.end(); ++it) {
				if (it.value() > max_votes) {
					max_votes = it.value();
					max_voted_label = it.key();
				}
			}

			label = max_voted_label;
		}

		return label;
	}

	DecisionTree::DecisionTree() {
	}

	void DecisionTree::construct(const std::vector<boost::shared_ptr<Example>>& examples, bool sample_attributes, int max_depth) {
		if (examples.size() == 0) return;

		root = constructNodes(examples, 0, sample_attributes, max_depth);

		root->setLabelFromChildren();
	}

	int DecisionTree::test(const boost::shared_ptr<Example>& example) {
		if (!root) throw "Tree is not constructed.";

		return root->test(example);
	}

	void DecisionTree::save(const QString& filename) {
		QFile file(filename);
		if (!file.open(QFile::WriteOnly)) throw "File cannot open.";

		QDomDocument doc;

		QDomElement root = save(doc);
		doc.appendChild(root);

		QTextStream out(&file);
		doc.save(out, 4);
	}

	QDomElement DecisionTree::save(QDomDocument& doc) {
		QDomElement tree_node = doc.createElement("tree");

		if (root) {
			QDomElement node = root->save(doc);
			tree_node.appendChild(node);
		}
		
		return tree_node;
	}

	boost::shared_ptr<DecisionTreeNode> DecisionTree::constructNodes(const std::vector<boost::shared_ptr<Example>>& examples, int depth, bool sample_attributes, int max_depth) {
		boost::shared_ptr<DecisionTreeNode> node = boost::shared_ptr<DecisionTreeNode>(new DecisionTreeNode(depth));

		// check if the labels are the same across the examples
		QMap<unsigned char, int> labels;
		for (int i = 0; i < examples.size(); ++i) {
			if (!labels.contains(examples[i]->label)) {
				labels[examples[i]->label] = 0;
			}
			labels[examples[i]->label]++;
		}
		if (labels.size() == 1) {
			// single label, and no need for futher splitting
			node->label = labels.begin().key();
			return node;
		}

		// check if the depth exceeds the max depth
		if (depth >= max_depth) {
			unsigned char max_voted_label = Example::LABEL_UNKNOWN;
			int max_votes = 0;
			for (auto it = labels.begin(); it != labels.end(); ++it) {
				if (it.value() > max_votes) {
					max_votes = it.value();
					max_voted_label = it.key();
				}
			}

			node->label = max_voted_label;
			return node;
		}

		// randomly sample the attributes
		std::vector<unsigned int> indices;
		if (sample_attributes) {
			indices = std::vector<unsigned int>(examples[0]->data.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::random_shuffle(indices.begin(), indices.end());
			indices.resize(sqrt(indices.size()));
		}
		else {
			indices = std::vector<unsigned int>(examples[0]->data.size());
			std::iota(std::begin(indices), std::end(indices), 0);
		}

		// find the best attribute to split
		float min_e = std::numeric_limits<float>::max();
		int best_attribute = -1;
		for (int i = 0; i < indices.size(); ++i) {
			float e = calculateEntropy(examples, indices[i]);
			if (e < min_e) {
				min_e = e;
				best_attribute = indices[i];
			}
		}
		node->split_attribute_id = best_attribute;

		// split the examples
		QMap<unsigned char, std::vector<boost::shared_ptr<Example>>> subsets;
		for (int i = 0; i < examples.size(); ++i) {
			unsigned char val = examples[i]->data[best_attribute];
			if (!subsets.contains(val)) {
				subsets[val] = std::vector<boost::shared_ptr<Example>>();
			}
			subsets[val].push_back(examples[i]);
		}

		for (auto it = subsets.begin(); it != subsets.end(); ++it) {
			boost::shared_ptr<DecisionTreeNode> child_node = constructNodes(it.value(), depth + 1, sample_attributes, max_depth);
			node->children[it.key()] = child_node;
		}

		return node;
	}

	float DecisionTree::calculateEntropy(const std::vector<boost::shared_ptr<Example>>& examples, int split_attribute) {
		// split the examples
		QMap<unsigned char, QMap<unsigned char, int>> histogram;
		QMap<unsigned char, int> count;
		for (int i = 0; i < examples.size(); ++i) {
			unsigned char val = examples[i]->data[split_attribute];
			unsigned char label = examples[i]->label;
			if (!histogram.contains(val)) {
				histogram[val] = QMap<unsigned char, int>();
				count[val] = 0;
			}
			if (!histogram[val].contains(label)) {
				histogram[val][label] = 0;
			}
			histogram[val][label]++;
			count[val]++;
		}

		// calculate the entropy
		float total_entropy = 0.0f;
		for (auto it = histogram.begin(); it != histogram.end(); ++it) {
			float entropy = 0.0f;
			if (it.value().size() <= 1) {
				entropy = 0.0f;
			}
			else {
				for (auto it2 = it.value().begin(); it2 != it.value().end(); ++it2) {
					float p = (float)(it2.value()) / count[it.key()];
					entropy -= p * std::log2(p);
				}
			}
			total_entropy += entropy * count[it.key()];
		}

		return total_entropy / examples.size();
	}
	
	RandomForest::RandomForest() {
	}

	void RandomForest::construct(const std::vector<boost::shared_ptr<Example>>& examples, int num_trees, float ratio, int max_depth) {
		trees.clear();

		for (int i = 0; i < num_trees; ++i) {
			printf("Tree: %d\n", i + 1);

			// randomly sample the examples
			std::vector<unsigned int> indices(examples.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::random_shuffle(indices.begin(), indices.end());
			indices.resize(examples.size() * ratio);

			std::vector<boost::shared_ptr<Example>> subset(indices.size());
			for (int j = 0; j < subset.size(); ++j) {
				subset[j] = examples[indices[j]];
			}
			
			// construct a decision tree
			DecisionTree dt;
			dt.construct(subset, true, max_depth);

			trees.push_back(dt);
		}
	}

	void RandomForest::save(const QString& filename) {
		QFile file(filename);
		if (!file.open(QFile::WriteOnly)) throw "File cannot open.";

		QDomDocument doc;

		// set root node
		QDomElement root = doc.createElement("random_forest");
		doc.appendChild(root);

		// write trees
		for (int i = 0; i < trees.size(); ++i) {
			QDomElement tree_node = trees[i].save(doc);			

			root.appendChild(tree_node);
		}

		QTextStream out(&file);
		doc.save(out, 4);
	}

	int RandomForest::test(const boost::shared_ptr<Example>& example) {
		if (trees.size() == 0) throw "Random forest is not constructed.";

		QMap<unsigned char, int> histogram;
		for (int i = 0; i < trees.size(); ++i) {
			unsigned char label = trees[i].test(example);
			if (!histogram.contains(label)) {
				histogram[label] = 0;
			}
			histogram[label]++;
		}

		// find the maximum vote
		int max_votes = 0;
		unsigned char max_voted_label = Example::LABEL_UNKNOWN;
		for (auto it = histogram.begin(); it != histogram.end(); ++it) {
			if (it.value() > max_votes) {
				max_votes = it.value();
				max_voted_label = it.key();
			}
		}

		return max_voted_label;
	}
}