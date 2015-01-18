//
//  main.cpp
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/6/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//

#define LOCAL true

#define USE_REGERESSION
//#define USE_XGBOOST

//#define USE_FEATURES_PRUNNING

#ifndef USE_REGERESSION
#include "WeightedKNN.h"
#endif

#ifdef USE_XGBOOST
#include "xgboost/io/io.h"
#include "xgboost/io/simple_dmatrix-inl.hpp"
#include "xgboost/utils/utils.h"
#include "xgboost/utils/config.h"
#include "xgboost/learner/learner-inl.hpp"
#include "xgboost/data.h"
#endif


#ifdef LOCAL
#include "stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include <iostream>
#include <sys/time.h>

using namespace std;


#define INLINE   inline __attribute__ ((always_inline))
#define NOINLINE __attribute__ ((noinline))

#define ALIGNED __attribute__ ((aligned(16)))

#define likely(x)   __builtin_expect(!!(x),1)
#define unlikely(x) __builtin_expect(!!(x),0)

#define SSELOAD(a)     _mm_load_si128((__m128i*)&a)
#define SSESTORE(a, b) _mm_store_si128((__m128i*)&a, b)

#define FOR(i,a,b)  for(int i=(a);i<(b);++i)
#define REP(i,a)    FOR(i,0,a)
#define ZERO(m)     memset(m,0,sizeof(m))
#define ALL(x)      x.begin(),x.end()
#define LL          long long
#define ULL         unsigned long long
#define LD          long double
#define MP          make_pair
#define VC          vector
#define PII         pair <int, int>
#define VI          VC < int >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VPII        VC < PII >
#define VD          VC < double >
#define VVD         VC < VD >
#define VF          VC < float >
#define VVF         VC < VF >
#define VS          VC < string >
#define VVS         VC < VS >

template<class T> void print(VC < T > v) {cerr << "[";if (v.size()) cerr << v[0];FOR(i, 1, v.size()) cerr << ", " << v[i];cerr << "]" << endl;}

template <class T>
void concatenate(VC<T> &first, const VC<T> &second) {
    size_t size = second.size();
    if (first.size() < size) {
        // resize
        first.resize(size);
    }
    // update
    for (int i = 0; i < size; i++) {
        first[i] += second[i];
    }
}

inline VS splt(string s, char c = ',') {
    VS all;
    int p = 0, np;
    while (np = s.find(c, p), np >= 0) {
        if (np != p)
            all.push_back(s.substr(p, np - p));
        else
            all.push_back("");
        p = np + 1;
    }
    if (p < s.size())
        all.push_back(s.substr(p));
    return all;
}

static bool LOG_DEBUG = true;
/*! the message buffer length */
const int kPrintBuffer = 1 << 12;
inline void Printf(const char *fmt, ...) {
    if (LOG_DEBUG) {
        std::string msg(kPrintBuffer, '\0');
        va_list args;
        va_start(args, fmt);
        vsnprintf(&msg[0], kPrintBuffer, fmt, args);
        va_end(args);
        fprintf(stderr, "%s", msg.c_str());
    }
}

inline void Assert(bool exp, const char *fmt, ...) {
    if (!exp) {
        std::string msg(kPrintBuffer, '\0');
        va_list args;
        va_start(args, fmt);
        vsnprintf(&msg[0], kPrintBuffer, fmt, args);
        va_end(args);
        fprintf(stderr, "AssertError:%s\n", msg.c_str());
        exit(-1);
    }
}


inline double getTime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

template<class bidiiter> bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        bidiiter r = begin;
        std::advance(r, rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

class RandomSample {
    // class members
    // generate "m_number" of data with the value within the range [0, m_max].
    int m_max;
    int m_number;
    
public:
    RandomSample(int max, int number) : m_max(max), m_number(number) {}
    
    inline VI get_sample_index() {
        // fill vector with indices
        VI re_res(m_max);
        for (int i = 0; i < m_max; ++i)
            re_res[i] = i;
        
        // suffle
        random_unique(re_res.begin(), re_res.end(), m_number);
        
        // resize vector
        re_res.resize(m_number);
        VI(re_res).swap(re_res);
        
        return re_res;
    }
};

struct Node {
    double m_node_value;
    int m_feature_index;
    double m_terminal_left;
    double m_terminal_right;
    
    // Each non-leaf node has a left child and a right child.
    Node *m_left_child = NULL;
    Node *m_right_child = NULL;
    
    // Construction function
    Node(double value, int feature_index, double value_left, double value_right) :
    m_node_value(value), m_feature_index(feature_index), m_terminal_left(value_left), m_terminal_right(value_right) {}
    
private:
    
    Node(Node const &); // non construction-copyable
    Node& operator=(Node const &); // non copyable
};

struct BestSplit {
    // the index of split feature
    int m_feature_index;
    // the calculated node value
    double m_node_value;
    // if false - split failed
    bool m_status;
    
    // construction function
    BestSplit() : m_feature_index(0.0), m_node_value(0.0), m_status(false) {}
};

struct SplitRes {
    VC<VD> m_feature_left;
    VC<VD> m_feature_right;
    double m_left_value = 0.0;
    double m_right_value = 0.0;
    VD m_obs_left;
    VD m_obs_right;
    
    // construction function
    SplitRes() : m_left_value(0.0), m_right_value(0.0) {}
};

struct ListData {
    double m_x;
    double m_y;
    
    ListData(double x, double y) : m_x(x), m_y(y) {}
    
    bool operator < (const ListData& str) const {
        return (m_x < str.m_x);
    }
};

typedef enum _TerminalType {
    AVERAGE, MAXIMAL
}TerminalType;

class RegressionTree {
private:
    // class members
    int m_min_nodes;
    int m_max_depth;
    int m_current_depth;
    TerminalType m_type;
    
public:
    // The root node
    Node *m_root = NULL;
    // The features importance per index
    VI features_importance;
    
    // construction function
    RegressionTree() : m_min_nodes(10), m_max_depth(3), m_current_depth(0), m_type(AVERAGE) {}
    
    // set parameters
    void setMinNodes(int min_nodes) {
        Assert(min_nodes > 3, "The number of terminal nodes is too small: %i", min_nodes);
        m_min_nodes = min_nodes;
    }
    
    void setDepth(int depth) {
        Assert(depth > 0, "Tree depth must be positive: %i", depth);
        m_max_depth = depth;
    }
    
    // get fit value
    double predict(const VD &feature_x) const{
        double re_res = 0.0;
        
        if (!m_root) {
            // failed in building the tree
            return re_res;
        }
        
        Node *current = m_root;
        
        while (true) {
            // current node information
            int c_feature_index = current->m_feature_index;
            double c_node_value = current->m_node_value;
            double c_node_left_value = current->m_terminal_left;
            double c_node_right_value = current->m_terminal_right;
            
            if (feature_x[c_feature_index] < c_node_value) {
                // we should consider left child
                current = current->m_left_child;
                
                if (!current) {
                    re_res = c_node_left_value;
                    break;
                }
            } else {
                // we should consider right child
                current = current->m_right_child;
                
                if (!current) {
                    re_res = c_node_right_value;
                    break;
                }
            }
        }
        
        return re_res;
    }
    
    /*
     *  The method to build regression tree
     */
    void buildRegressionTree(const VC<VD> &feature_x, const VD &obs_y) {
        int samples_num = feature_x.size();
        
        Assert(samples_num == obs_y.size() && samples_num != 0,
               "The number of samles does not match with the number of observations or the samples number is 0. Samples: %i", samples_num);
        
        Assert (m_min_nodes * 2 <= samples_num, "The number of samples is too small");
        
        int feature_dim = feature_x[0].size();
        features_importance.resize(feature_dim, 0);
        
        // build the regression tree
        buildTree(feature_x, obs_y);
    }
    
private:
    
    /*
     *  The following function gets the best split given the data
     */
    BestSplit findOptimalSplit(const VC<VD> &feature_x, const VD &obs_y) {
        
        BestSplit split_point;
        
        if (m_current_depth > m_max_depth) {
            return split_point;
        }
        
        int samples_num = feature_x.size();
        
        if (m_min_nodes * 2 > samples_num) {
            // the number of observations in terminals is too small
            return split_point;
        }
        int feature_dim = feature_x[0].size();
        
        
        double min_err = 0;
        int split_index = -1;
        double node_value = 0.0;
        
        // begin to get the best split information
        for (int loop_i = 0; loop_i < feature_dim; loop_i++){
            // get the optimal split for the loop_index feature
            
            // get data sorted by the loop_i-th feature
            VC<ListData> list_feature;
            for (int loop_j = 0; loop_j < samples_num; loop_j++) {
                list_feature.push_back(ListData(feature_x[loop_j][loop_i], obs_y[loop_j]));
            }
            
            // sort the list
            sort(list_feature.begin(), list_feature.end());
            
            // begin to split
            double sum_left = 0.0;
            double mean_left = 0.0;
            int count_left = 0;
            double sum_right = 0.0;
            double mean_right = 0.0;
            int count_right = 0;
            double current_node_value = 0;
            double current_err = 0.0;
            
            // initialize left
            for (int loop_j = 0; loop_j < m_min_nodes; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                sum_left += fetched_data.m_y;
                count_left++;
            }
            mean_left = sum_left / count_left;
            
            // initialize right
            for (int loop_j = m_min_nodes; loop_j < samples_num; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                sum_right += fetched_data.m_y;
                count_right++;
            }
            mean_right = sum_right / count_right;
            
            // calculate the current error
            // err = ||x_l - mean(x_l)||_2^2 + ||x_r - mean(x_r)||_2^2
            // = ||x||_2^2 - left_count * mean(x_l)^2 - right_count * mean(x_r)^2
            // = constant - left_count * mean(x_l)^2 - right_count * mean(x_r)^2
            // Thus, we only need to check "- left_count * mean(x_l)^2 - right_count * mean(x_r)^2"
            current_err = -1 * count_left * mean_left * mean_left - count_right * mean_right * mean_right;
            
            // current node value
            current_node_value = (list_feature[m_min_nodes].m_x + list_feature[m_min_nodes - 1].m_x) / 2;
            
            if (current_err < min_err && current_node_value != list_feature[m_min_nodes - 1].m_x) {
                split_index = loop_i;
                node_value = current_node_value;
                min_err = current_err;
            }
            
            // begin to find the best split point for the feature
            for (int loop_j = m_min_nodes; loop_j <= samples_num - m_min_nodes - 1; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                double y = fetched_data.m_y;
                sum_left += y;
                count_left++;
                mean_left = sum_left / count_left;
                
                
                sum_right -= y;
                count_right--;
                mean_right = sum_right / count_right;
                
                
                current_err = -1 * count_left * mean_left * mean_left - count_right * mean_right * mean_right;
                // current node value
                current_node_value = (list_feature[loop_j + 1].m_x + fetched_data.m_x) / 2;
                
                if (current_err < min_err && (current_node_value != fetched_data.m_x)) {
                    split_index = loop_i;
                    node_value = current_node_value;
                    min_err = current_err;
                }
                
            }
        }
        // set the optimal split point
        if (split_index == -1) {
            // failed to split data
            return split_point;
        }
        split_point.m_feature_index = split_index;
        split_point.m_node_value = node_value;
        split_point.m_status = true;
        
        return split_point;
    }
    
    /*
     *  Split data into the left node and the right node based on the best splitting
     *  point.
     */
    SplitRes splitData(const VC<VD> &feature_x, const VD &obs_y, const BestSplit &best_split) {
        
        SplitRes split_res;
        
        int feature_index = best_split.m_feature_index;
        double node_value = best_split.m_node_value;
        
        int samples_count = obs_y.size();
        for (int loop_i = 0; loop_i < samples_count; loop_i++) {
            VD ith_feature = feature_x[loop_i];
            if (ith_feature[feature_index] < node_value) {
                // append to the left feature
                split_res.m_feature_left.push_back(ith_feature);
                // observation
                split_res.m_obs_left.push_back(obs_y[loop_i]);
            } else {
                // append to the right
                split_res.m_feature_right.push_back(ith_feature);
                split_res.m_obs_right.push_back(obs_y[loop_i]);
            }
        }
        
        // update terminal values
        if (m_type == AVERAGE) {
            double mean_value = 0.0;
            for (double obsL : split_res.m_obs_left) {
                mean_value += obsL;
            }
            mean_value = mean_value / split_res.m_obs_left.size();
            split_res.m_left_value = mean_value;
            
            mean_value = 0.0;
            for (double obsR : split_res.m_obs_right) {
                mean_value += obsR;
            }
            mean_value = mean_value / split_res.m_obs_right.size();
            split_res.m_right_value = mean_value;
            
        } else if (m_type == MAXIMAL) {
            double max_value = 0.0;
            VD::iterator iter = split_res.m_obs_left.begin();
            if (++iter != split_res.m_obs_left.end()) {
                max_value = *iter;
            }
            
            while (++iter != split_res.m_obs_left.end()) {
                double sel_value = *iter;
                if (max_value < sel_value) {
                    max_value = sel_value;
                }
            }
            
            split_res.m_left_value = max_value;
            
            
            // right value
            max_value = 0.0;
            iter = split_res.m_obs_right.begin();
            if (++iter != split_res.m_obs_right.end()) {
                max_value = *iter;
            }
            
            while (++iter != split_res.m_obs_right.end()) {
                double sel_value = *iter;
                if (max_value < sel_value) {
                    max_value = sel_value;
                }
            }
            
            split_res.m_right_value = max_value;
            
        } else {
            // Unknown terminal type
            assert(false);
        }
        
        // return the result
        return split_res;
    }
    
    /*
     *  The following function builds a regression tree from data
     */
    Node* buildTree(const VC<VD> &feature_x, const VD &obs_y) {
        
        // obtain the optimal split point
        m_current_depth = m_current_depth + 1;
        
        BestSplit best_split = findOptimalSplit(feature_x, obs_y);
        
        if (!best_split.m_status) {
            if (m_current_depth > 0)
                m_current_depth = m_current_depth - 1;
            
            return NULL;
        }
        
        // update feature importance info
        features_importance[best_split.m_feature_index] += 1;
        
        // split the data
        SplitRes split_data = splitData(feature_x, obs_y, best_split);
        
        // append current value to tree
        Node *new_node = new Node(best_split.m_node_value, best_split.m_feature_index, split_data.m_left_value, split_data.m_right_value);
        
        if (!m_root) {
            m_root = new_node;
            m_current_depth = 0;
            // append left and right side
            m_root->m_left_child = buildTree(split_data.m_feature_left, split_data.m_obs_left); // left
            m_root->m_right_child = buildTree(split_data.m_feature_right, split_data.m_obs_right); // right
        } else {
            // append left and right side
            new_node->m_left_child = buildTree(split_data.m_feature_left, split_data.m_obs_left); // left
            new_node->m_right_child = buildTree(split_data.m_feature_right, split_data.m_obs_right); // right
        }
        if (m_current_depth > 0)
            m_current_depth--;
        
        return new_node;
    }
};

class PredictionForest {
public:
    // class members
    double m_init_value;
    // the tree forest
    VC<RegressionTree> m_trees;
    double m_combine_weight;
    
    // construction function
    PredictionForest(double learning_rate) : m_init_value(0.0), m_combine_weight(learning_rate) {}
    
    /**
     * The method to make prediction for estimate of function's value from provided features
     *
     * @param feature_x the features to use for prediction
     * @return the estimated function's value
     */
    double predict(const VD &feature_x) {
        double re_res = m_init_value;
        
        if (m_trees.size() == 0) {
            return re_res;
        }
        
        for (int i = 0; i < m_trees.size(); i++) {
            re_res += m_combine_weight * m_trees[i].predict(feature_x);
        }
        
        return re_res;
    }
    
    /**
     * Calculates importance of each feature in input samples
     */
    VI featureImportances() {
        VI importances;
        for (int i = 0; i < m_trees.size(); i++) {
            concatenate(importances, m_trees[i].features_importance);
        }
        return importances;
    }
};


class GradientBoostingTree {
    // class members
    double m_sampling_size_ratio = 0.5;
    double m_learning_rate = 0.01;
    int m_tree_number = 100;
    
    // tree related parameters
    int m_tree_min_nodes = 10;
    int m_tree_depth = 3;
    
public:
    
    GradientBoostingTree(double sample_size_ratio, double learning_rate,
                         int tree_number, int tree_min_nodes, int tree_depth) :
    m_sampling_size_ratio(sample_size_ratio), m_learning_rate(learning_rate), m_tree_number(tree_number),
    m_tree_min_nodes(tree_min_nodes), m_tree_depth(tree_depth) {
        // Check the validity of numbers
        Assert(sample_size_ratio > 0 && learning_rate > 0 && tree_number > 0 && tree_min_nodes >= 3 && tree_depth > 0,
               "Wrong parameters");
    }
    
    /**
     * Fits a regression function using the Gradient Boosting Tree method.
     * On success, return function; otherwise, return null.
     *
     * @param input_x the input features
     * @param input_y the ground truth values - one per features row
     */
    PredictionForest *fitGradientBoostingTree(const VC<VD> &input_x, const VD &input_y) {
        
        // initialize forest
        PredictionForest *res_fun = new PredictionForest(m_learning_rate);
        
        // get the feature dimension
        int feature_num = input_y.size();
        
        Assert(feature_num == input_x.size() && feature_num > 0,
               "Error: The input_x size should not be zero and should match the size of input_y");
        
        // get an initial guess of the function
        double mean_y = 0.0;
        for (double d : input_y) {
            mean_y += d;
        }
        mean_y = mean_y / feature_num;
        res_fun->m_init_value = mean_y;
        
        
        // prepare the iteration
        VD h_value(feature_num);
        // initialize h_value
        int index = 0;
        while (index < feature_num) {
            h_value[index] = mean_y;
            index += 1;
        }
        
        // begin the boosting process
        int iter_index = 0;
        while (iter_index < m_tree_number) {
            
            // calculate the gradient
            VD gradient;
            index = 0;
            for (double d : input_y) {
                gradient.push_back(d - h_value[index]);
                
                // next
                index++;
            }
            
            // begin to sample
            if (m_sampling_size_ratio < 0.99) {
                // sample without replacement
                
                // we need to sample
                RandomSample sampler(feature_num, (int) (m_sampling_size_ratio * feature_num));
                
                // get random index
                VI sampled_index = sampler.get_sample_index();
                
                // data for growing trees
                VC<VD> train_x;
                VD train_y;
                
                for (int sel_index : sampled_index) {
                    // assign value
                    train_y.push_back(gradient[sel_index]);
                    train_x.push_back(input_x[sel_index]);
                }
                
                // fit a regression tree
                RegressionTree tree;
                
                if (m_tree_depth > 0) {
                    tree.setDepth(m_tree_depth);
                }
                
                if (m_tree_min_nodes > 0) {
                    tree.setMinNodes(m_tree_min_nodes);
                }
                
                tree.buildRegressionTree(train_x, train_y);
                
                // store tree information
                if (tree.m_root == NULL) {
                    // clear buffer
                    train_x.clear();
                    train_y.clear();
                    continue;
                }
                
                res_fun->m_trees.push_back(tree);
                
                // update h_value information, prepare for the next iteration
                int sel_index = 0;
                while (sel_index < feature_num) {
                    h_value[sel_index] += m_learning_rate * tree.predict(input_x[sel_index]);
                    sel_index++;
                }
                
            } else {
                // use all data
                // fit a regression tree
                RegressionTree tree;
                
                // set parameters if needed
                if (m_tree_depth > 0) {
                    tree.setDepth(m_tree_depth);
                }
                
                if (m_tree_min_nodes > 0) {
                    tree.setMinNodes(m_tree_min_nodes);
                }
                
                tree.buildRegressionTree(input_x, gradient);
                
                if (tree.m_root == NULL) {
                    // cannot update any more
                    break;
                }
                // store tree information
                res_fun->m_trees.push_back(tree);
                
                // update h_value information, prepare for the next iteration
                for (int loop_index = 0; loop_index < feature_num; loop_index++) {
                    h_value[loop_index] += m_learning_rate * tree.predict(input_x[loop_index]);
                }
            }
            
            // next iteration
            iter_index++;
        }
        
        // set the learning rate and return
        // res_fun.m_combine_weight = m_learning_rate;
        
        return res_fun;
    }
    
    PredictionForest *learnGradientBoostingRanker(const VC<VD> &input_x, const VC<VD> &input_y, const double tau) {
        PredictionForest *res_fun = new PredictionForest(m_learning_rate);
        
        int feature_num = input_x.size();
        
        Assert(feature_num == input_y.size() && feature_num > 0,
               "The size of input_x should be the same as the size of input_y");
        
        VD h_value_x(feature_num, 0);
        VD h_value_y(feature_num, 0);
        
        int iter_index = 0;
        while (iter_index < m_tree_number) {
            
            // in the boosting ranker, randomly select half samples without replacement in each iteration
            RandomSample sampler(feature_num, (int) (0.5 * feature_num));
            
            // get random index
            VI sampled_index = sampler.get_sample_index();
            
            VC<VD> gradient_x;
            VD gradient_y;
            
            for (int i = 0; i < sampled_index.size(); i++) {
                int sel_index = sampled_index[i];
                
                gradient_x.push_back(input_x[sel_index]);
                gradient_x.push_back(input_y[sel_index]);
                
                // get sample data
                if (h_value_x[sel_index] < h_value_y[sel_index] + tau) {
                    double neg_gradient = h_value_y[sel_index] + tau - h_value_x[sel_index];
                    gradient_y.push_back(neg_gradient);
                    gradient_y.push_back(-1 * neg_gradient);
                } else {
                    gradient_y.push_back(0.0);
                    gradient_y.push_back(0.0);
                }
                //                cerr << "sel_index: " << sel_index << endl;
            }
            
            // fit a regression tree
            RegressionTree tree;
            //            tree.m_type = MAXIMAL;
            
            tree.buildRegressionTree(gradient_x, gradient_y);
            
            // store tree information
            if (tree.m_root == NULL) {
                continue;
            }
            
            // update information
            res_fun->m_trees.push_back(tree);
            
            double err = 0.0;
            
            for (int loop_index = 0; loop_index < feature_num; loop_index++) {
                h_value_x[loop_index] += m_learning_rate * tree.predict(input_x[loop_index]);
                h_value_y[loop_index] += m_learning_rate * tree.predict(input_y[loop_index]);
                
                if (h_value_x[loop_index] < h_value_y[loop_index] + tau) {
                    err += (h_value_x[loop_index] - h_value_y[loop_index] - tau) *
                    (h_value_x[loop_index] - h_value_y[loop_index] - tau);
                }
            }
            //            if (LOG_DEBUG) cerr << iter_index + 1 << "-th iteration with error " << err << endl;
            
            iter_index += 1;
        }
        
        
        
        return res_fun;
    }
};

//
// ------------------------------------------------------
//

struct Entry {
    static constexpr double NVL = -1000;
    
    // The number of features
    static constexpr int fCount = 22;
    
    int id;
    VD features;
    double dv;
    std::string formula;
    bool completeFeatures = true;
    VI missedFeatures;
    
    bool important[fCount];
    
    inline static double parse(const std::string &s) {
        return s == "" ? NVL : atof(s.c_str());
    }
    
    Entry(int sId) {
        id = sId;
        initImportanceVector();
    }
    
    Entry(int sId, const std::string &s) {
        id = sId;
        initImportanceVector();
        VS vs = splt(s, ',');
        int index = 0;
        REP(i, fCount) {
            if (i == 14) {
                formula = vs[i];
            } else {
                double f = parse(vs[i]);
                // check if feature marked as important
                if (important[index]) {
                    features.push_back(f);
                    if (f == NVL) {
                        completeFeatures = false;
                        // store missed feature
                        missedFeatures.push_back(index);
                    }
                }
                index++;
            }
        }
        
        // set DV if present
        if (vs.size() == fCount + 1) {
            dv = parse(vs[fCount]);
            if (dv < 0) {
                // missed activity
                dv = NVL;
            }
        } else {
            dv = NVL;
        }
    }
    
    inline bool operator < (const Entry& entry) const {
        return (dv < entry.dv);
    }
    
private:
    inline void initImportanceVector() {
#ifdef USE_FEATURES_PRUNNING
        //        int indexes[] = {2, 7, 16, 18, 12, 11, 20, 1, 5}; // 902627.26
        //        int indexes[] = {3, 2, 9, 16, 12, 7, 6, 18, 11, 8, 1, 20, 5, 0, 14, 15, 17}; // 910678.93
        //        int indexes[] = {2, 9, 16, 12, 7, 6, 18, 11, 8, 1, 20, 5, 0, 14, 15}; // 910676.63
        //        int indexes[] = {2, 9, 7, 5, 16, 8, 11, 18, 12, 1, 14, 20, 6, 10, 15, 17, 19}; // 902973.90
        //        int indexes[] = {7, 2, 8, 9, 12, 6, 5, 16, 15, 14, 17, 10, 20, 11, 18, 0, 19, 1, 4, 3}; // 893954.08
        
        int indexes[] = {1, 2, 7, 9, 12, 5, 16, 8, 6, 18, 20, 14, 15, 11, 17, 0, 10, 19, 4}; // 901736.64
        
        for (int i = 0; i < fCount; i++) important[i] = false;
        int sizeIn = (sizeof(indexes)/sizeof(*indexes));
        for (int i = 0; i < sizeIn; i++) {
            important[indexes[i]] = true;
        }
#else
        for (int i = 0; i < fCount; i++) important[i] = true;
#endif
    }
};

struct GBTConfig {
    double sampling_size_ratio = 0.5;
    double learning_rate = 0.01;
    int tree_number = 130;
    int tree_min_nodes = 10;
    int tree_depth = 3;
};

inline void parseData(const VS &data, const int startIndex, VC<Entry> &v) {
    int index = startIndex;
    for (const string &s : data) {
        v.push_back(Entry(index++, s));
    }
}

inline void imputation(VC<Entry> &entries) {
    VD vals(Entry::fCount, 0);
    VD counts(Entry::fCount, 0);
    int size = entries.size();
    // calculate mean for all features per feature
    for (int i = 0 ; i < size; i++) {
        for (int j = 0; j < entries[i].features.size(); j++) {
            if (entries[i].features[j] != Entry::NVL) {
                vals[j] += entries[i].features[j];
                counts[j] += 1;
            }
        }
    }
    for (int i = 0; i < vals.size(); i++) {
        if (vals[i] && counts[i]) {
            vals[i] = vals[i] / counts[i];
        }
    }
    if (LOG_DEBUG) {
        cerr << "Means =========" << endl; print<double>(vals);
        cerr << "Counts =========" << endl; print<double>(counts);
    }
    
    // correct missing features
    int samplesCorrected = 0;
    for (int i = 0 ; i < size; i++) {
        if (!entries[i].completeFeatures) {
            VI &missed = entries[i].missedFeatures;
            for (int j = 0; j < missed.size(); j++) {
                int missedIndex = missed[j];
                entries[i].features[missedIndex] = vals[missedIndex];
            }
            entries[i].completeFeatures = true;
            samplesCorrected++;
        }
    }
    Printf("Samples corrected: %i\n", samplesCorrected);
};

inline void imputation(VC<Entry>&training, VC<Entry>&testing) {
    int X = training.size();
    int Y = testing.size();
    // join vectors
    VC<Entry> fullSet;
    fullSet.reserve( X + Y ); // preallocate memory
    fullSet.insert( fullSet.end(), training.begin(), training.end() );
    fullSet.insert( fullSet.end(), testing.begin(), testing.end() );
    
    // features imputation to correct missed values
    imputation(fullSet);
    
    // split vectors
    training.erase(training.begin(), training.end());
    training.insert(training.begin(), fullSet.begin(), fullSet.begin() + X);
    
    testing.erase(testing.begin(), testing.end());
    testing.insert(testing.begin(), fullSet.begin() + X, fullSet.end());
};

inline pair<int, int> correctZeroDVBySimilarity(const VC<VD> &matrix, VC<Entry> &training) {
    int X = training.size();
    int count = 0;
    int missedFeaturesSamples = 0;
    for (int i = 0; i < X; i++) {
        if (training[i].dv == Entry::NVL || training[i].dv == 0) {
            // set entry DV using its similarity
            double dvSum = 0;
            double wSum = 0;
            
            // go through the row of similar entries
            for (int j = 0; j < X; j++) {
                double sim = matrix[training[i].id][training[j].id];
                if (training[j].dv != Entry::NVL && training[i].id != training[j].id) {
                    dvSum += sim * training[j].dv;
                    wSum += sim;
                }
            }
            training[i].dv = dvSum / wSum;
            //            if (LOG_DEBUG) cerr << "Id: " << training[i].id << ", DV: " << training[i].dv << ", dvSum: " << dvSum << ", wSum:" << wSum << endl;
            
            count++;
        }
        
        if (!training[i].completeFeatures) {
            missedFeaturesSamples++;
        }
        
    }
    return pair<int, int>(count, missedFeaturesSamples);
}

// X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
// X_scaled = X_std * (max - min) + min
inline void scaleMinMax(VC<Entry> &entries, double min, double max) {
    int size = entries.size();
    // fin min/max per sample per feature
    VD fMins(size, 10000), fMaxs(size, -10000);
    for (int i = 0 ; i < size; i++) {
        for (int j = 0; j < entries[i].features.size(); j++) {
            if (fMaxs[j] < entries[i].features[j]) {
                fMaxs[j] = entries[i].features[j];
            }
            if (fMins[j] > entries[i].features[j]) {
                fMins[j] = entries[i].features[j];
            }
        }
    }
    
    // find X scaled
    for (int i = 0 ; i < size; i++) {
        for (int j = 0; j < entries[i].features.size(); j++) {
            double X = entries[i].features[j];
            double X_min = fMins[j];
            double X_max = fMaxs[j];
            double X_std = (X - X_min) / (X_max - X_min);
            double X_scaled = X_std * (max - min) + min;
            entries[i].features[j] = X_scaled;
        }
        cerr << "Entry id: " << entries[i].id << " "; print(entries[i].features);
    }
}
// X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
// X_scaled = X_std * (max - min) + min
inline void scaleDVMinMax(VC<Entry> &entries, double min, double max) {
    int size = entries.size();
    double Y_min = 10000;
    double Y_max = -10000;
    // fin min/max per sample
    for (int i = 0 ; i < size; i++) {
        double val = entries[i].dv != Entry::NVL ? entries[i].dv : 0;
        if (Y_max < val) {
            Y_max = val;
        }
        if (Y_min > val) {
            Y_min = val;
        }
    }
    Printf("Y_min: %f, Y_max: %f\n", Y_min, Y_max);
    
    // find Y scaled
    for (int i = 0 ; i < size; i++) {
        double Y = entries[i].dv != Entry::NVL ? entries[i].dv : 0;
        double Y_std = (Y - Y_min) / (Y_max - Y_min);
        double Y_scaled = Y_std * (max - min) + min;
        entries[i].dv = Y_scaled;
        
        Printf("Entry id: %i, dv: %f\n", entries[i].id, entries[i].dv);
    }
}

inline VD createClassificationLabels(VC<Entry> entries) {
    VD labels;
    int X = entries.size();
    int lIndex = 0;
    for (int i = 0; i < X; i++) {
        if (entries[i].dv != Entry::NVL) {
            bool labelSet = false;
            for (int j = 0; j < labels.size(); j++) {
                // check if value already has label
                if (labels[j] == entries[i].dv) {
                    entries[i].dv = j;
                    labelSet = true;
                    break;
                }
            }
            if (!labelSet) {
                // assign new label to the DV and store last value
                labels.push_back(entries[i].dv);
                entries[i].dv = lIndex;
                lIndex++;
            }
        }
    }
    return labels;
}

#ifndef USE_REGERESSION
inline void readExamples(const VC<Entry> &entries, TRAINING_EXAMPLES_LIST *rlist, bool test) {
    int X = entries.size();
    for (int i = 0; i < X; i++) {
        if (test || entries[i].dv != Entry::NVL) {
            TrainingExample *example = new TrainingExample();
            // copy features
            for (int j = 0; j < entries[i].features.size(); j++) {
                double val = entries[i].features[j];
                example->Value[j] = val;
            }
            // copy DV
            example->Value[NO_OF_ATT - 1] = entries[i].dv;
            
            // Generating random weights for instances.
            // These weights are used in instance WKNN
            double rno = (double)(rand () % 100 + 1);
            example->Weight = rno / 100;
            example->index = entries[i].id;
            example->isNearest2AtleastSome = false;
            
            //            cerr << "Entry id: " << entries[i].id << ", features: ";print(entries[i].features);
            
            printExample(*example);
            
            // add to the list
            rlist->insert (rlist->end(), *example);
            
            delete example;
        }
    }
}
#endif

#ifdef USE_XGBOOST
inline void readParamMatrix(const VC<Entry> &entries, xgboost::io::DMatrixSimple &dMatrix) {
    int entriesSize = entries.size();
    cerr << entriesSize << endl;
    std::vector<xgboost::RowBatch::Entry> feats;
    for (int i = 0; i < entriesSize; i++) {
        // iterate over features and collect in entry
        for (int j = 0; j < entries[i].features.size(); j++) {
            float val = entries[i].features[j];
            if (val != Entry::NVL) {
                feats.push_back(xgboost::RowBatch::Entry(j ,val));
            }
        }
        
        // collect DV
        dMatrix.info.labels.push_back((float)entries[i].dv);
        // add row batch
        dMatrix.AddRow(feats);
        
        // clear for next loop
        feats.clear();
    }
    
    long rowsSize = dMatrix.info.info.num_row;
    xgboost::utils::Assert(rowsSize == entriesSize, "Wrong batch entry rows collected. Expected: %d, found %d", rowsSize, entriesSize);
}
#endif

class ActiveMolecules {
    int X;
    int Y;
    int M;
    VC<VD>matrix;
    
public:
    
    int similarity(int moleculeID, VD &similarities) {
        if (!matrix.size()) {
            M = matrix.size();
        }
        VD row;
        for (double &d : similarities) {
            row.push_back(d);
        }
        matrix.push_back(row);
        
        return 0;
    }
    
    VI rank(const VS &train, const VS &test) {
        X = train.size();
        Y = test.size();
        
        Printf( "GB SC Training length: %i, testing length: %i\n", X, Y);
        
        // parse data
        VC<Entry> training;
        VC<Entry> testing;
        parseData(train, 0, training);
        parseData(test, X, testing);
        
        // features imputation to correct missed values
        imputation(training, testing);
        
        assert(training.size() == X && testing.size() == Y);
        
#ifdef USE_REGERESSION
#ifdef USE_XGBOOST
        // rank by XGBT regression
        VI res = renkByXGBRegression(training, testing);
#else
        // rank by GBT regression
        VI res = rankByGBTRegression(training, testing);
#endif
#else
        // rank by classification
        VI res = rankByClassification(training, testing);
        
#endif
        return res;
    }
    
private:
#ifdef USE_XGBOOST
    VI renkByXGBRegression(VC<Entry> &training, VC<Entry> &testing) {
        Printf("=========== Rank by XGBoost regression ===========\n");
        
        //
        // correct missed DV in training data
        //
        pair<int, int> correctedPair = correctZeroDVBySimilarity(matrix, training);
        
        Printf("Corrected: %i, found samples with missed features: %i\n", correctedPair.first, correctedPair.second);
        
        double startTime = getTime();
        
        // read data
        xgboost::io::DMatrixSimple trainMat;
        readParamMatrix(training, trainMat);
        
        xgboost::io::DMatrixSimple testMat;
        readParamMatrix(testing, testMat);
        
        //
        // initialize learner
        //
        xgboost::learner::BoostLearner learner;
        // step size shrinkage used in update to prevents overfitting.  After each boosting step, we can directly get the weights of new features.
        // And eta actually shrinkage the feature weights to make the boosting process more conservative. Learning rate.
        learner.SetParam("eta", "0.3");
        // minimum loss reduction required to make a further partition
        learner.SetParam("gamma", "3.0");
        // maximum depth of a tree
        learner.SetParam("max_depth", "3");
        // minimum sum of instance weight(hessian) needed in a child
        learner.SetParam("min_child_weight", "1");
        // subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
        learner.SetParam("subsample", "1");
        // evaluation metrics for validation data (rmse, logloss)
        learner.SetParam("eval_metric", "rmse");
        // the objective function (reg:linear, reg:logistic)
        learner.SetParam("objective", "reg:linear");
        
        learner.SetParam("silent", "1");
        
        
        Printf("Model configuration\n");
        
        // print configs
        std::vector< std::pair<std::string, std::string> > cfg = learner.cfg_;
        for (int i = 0; i < cfg.size(); i++) {
            Printf("%s=%s\n", cfg[i].first.c_str(), cfg[i].second.c_str());
        }

        
        // set cache data
        std::vector<xgboost::io::DataMatrix*> mats;
        mats.push_back(&trainMat);
        learner.SetCacheData(mats);
        
        // initialize model
        learner.InitModel();
        
        //        xgboost::utils::FeatMap featmap;
        //        std::vector<std::string> model_dump = learner.DumpModel(featmap, 1);
        //        for (size_t i = 0; i < model_dump.size(); ++i) {
        //            cerr << model_dump[i] << endl;
        //        }
        
        /*! \brief the names of the evaluation data used in output log */
        std::vector<std::string> eval_data_names;
        std::vector<const xgboost::io::DataMatrix*> devalall;
        devalall.push_back(&trainMat);
        eval_data_names.push_back(std::string("train"));
        //        devalall.push_back(&testMat);
        //        eval_data_names.push_back(std::string("test"));
        
        
        
        
//        std::vector<float> &vec =
//        static_cast<DataMatrix*>(handle)->info.GetFloatInfo(field);
//        vec.resize(len);
//        memcpy(BeginPtr(vec), info, sizeof(float) * len);


        
        
        //
        // Train learner
        //
        
        // number of boosting iterations
        int num_round = 100;
        
        const time_t start = time(NULL);
        unsigned long elapsed = 0;
        
        learner.CheckInit(&trainMat);
        
        //
        // Make base prediction
        //
        // train xgboost for 1 round
//        for (int i = 0; i < num_round; ++i) {
//            learner.UpdateOneIter(i, trainMat);
//            std::string res = learner.EvalOneIter(i, devalall, eval_data_names);
//        }
//        VC<float> ptrain;
//        learner.Predict(trainMat, true, &ptrain);
//        VC<float> ptest;
//        learner.Predict(testMat, true, &ptest);
//        
//        // set base margin
//        trainMat.info.base_margin = ptrain;
//        testMat.info.base_margin = ptest;
//        
//        cerr << "Base prediction finished" << endl;
        
        for (int i = 0; i < num_round; ++i) {
            
            elapsed = (unsigned long)(time(NULL) - start);
            Printf("boosting round %d, %lu sec elapsed\n", i, elapsed);
            
            learner.UpdateOneIter(i, trainMat);
            std::string res = learner.EvalOneIter(i, devalall, eval_data_names);
            
            Printf("%s\n ", res.c_str());
            elapsed = (unsigned long)(time(NULL) - start);
        }
        
        xgboost::utils::FeatMap fmap;
        std::vector<std::string> model = learner.DumpModel(fmap, 1);
        Printf("Dumped model size: %lu\n", model.size());
//        for (int i = 0; i < model.size(); i++) {
//            xgboost::utils::Printf("%s\n ", model[i].c_str());
//        }
        
        double rankTime = getTime();
        
        //
        // Predict
        //
        VC<float>Y_test;
        learner.Predict(testMat, false, &Y_test);
        
        // assign results
        for (int i = 0; i < Y; i++) {
            testing[i].dv = Y_test[i];
        }
        
        // sort
        sort(testing.rbegin(), testing.rend());
        
        
        VI ids;
        for (int i = 0; i < Y; i++) {
            Printf("ID: %i, activity: %f\n", testing[i].id, testing[i].dv);
            
            ids.push_back(testing[i].id);
        }
        
        double finishTime = getTime();
        
        Printf("Train time: %.2f, rank time: %.2f, full time: .2f\n", rankTime - startTime,  finishTime - rankTime, finishTime - startTime);
        
        return ids;
    }
#endif
    
#ifndef USE_REGERESSION
    VI rankByClassification(VC<Entry> &training, VC<Entry> &testing) {
        
        Printf("=========== Rank by classification ===========\n");
        
        double startTime = getTime();
        
        // make classification labels
        VD labels = createClassificationLabels(training);
        if (LOG_DEBUG) {
            Printf("Labels ++++++++++++++++");
            print<double>(labels);
            Printf("Labels size: %i\n", labels.size());
        }
        
        // prepare WNN data
        // Training Examples
        TRAINING_EXAMPLES_LIST elist;
        // Testing Examples
        TRAINING_EXAMPLES_LIST qlist;
        readExamples(training, &elist, false);
        readExamples(testing, &qlist, true);
        
        int knnTrainingSize = elist.size();
        int knnTestingSize = qlist.size();
        cerr << "Real training size: " << knnTrainingSize << ", testing size: " << knnTestingSize << endl;
        
        // Normalize values using standard deviation
        //        NormalizeByStandardDeviation (&elist, knnTrainingSize);
        //        NormalizeByStandardDeviation(&qlist, knnTestingSize);
        
        // run WNN algorithms
        VI testClasses = classifyByKNNBackwardElimination(&elist, &qlist);
        
        double rankTime = getTime();
        
        if (LOG_DEBUG)cerr << "Test classes: "; print(testClasses);
        
        // put classes into labels
        for (int i = 0; i < testing.size(); i++) {
            int classOfTest = testClasses[i];
            testing[i].dv = labels[classOfTest];
        }
        double finishTime = getTime();
        
        if (LOG_DEBUG) cerr << "++++ OUT ++++" << endl;
        
        // sort to have highest rating at the top
        sort(testing.rbegin(), testing.rend());
        
        VI ids;
        for (int i = 0; i < Y; i++) {
            if (LOG_DEBUG) cerr << "ID: " << testing[i].id << ", activity: " << testing[i].dv << endl;
            
            ids.push_back(testing[i].id);
        }
        
        cerr << "Rank time: " << rankTime - startTime << ", full time: " << finishTime - startTime << endl;
        
        return ids;
        
    }
#endif
    
    VI rankByGBTRegression(VC<Entry> &training, VC<Entry> &testing) {
        cerr << "=========== Rank by GBT regression ===========" << endl;
        
        //
        // normalize features
        //
        //        scaleMinMax(training, 0, 1);
        //        scaleMinMax(testing, 0, 1);
        
        //
        // Normalize DVs in training
        //
        //        scaleDVMinMax(training, 0, 1);
        
        
        //
        // correct missed DV in training data
        //
        pair<int, int> correctedPair = correctZeroDVBySimilarity(matrix, training);
        cerr << "Corrected: " << correctedPair.first << ", found samples with missed features: " << correctedPair.second << endl;
        
        // prepare data
        VC<VD> input_x;
        VD input_y;
        for (int i = 0; i < X; i++) {
            if (training[i].dv != Entry::NVL) {
                // check for DV set only for training data and add TEST data without checks
                input_x.push_back(training[i].features);
                input_y.push_back(training[i].dv);
            } else if (LOG_DEBUG){
                //                cerr << "Entry id: " << e.id << " missing DV" << endl;
            }
        }
        
        if (LOG_DEBUG) cerr << "Real train size: " << input_x.size() << endl;
        
        double startTime = getTime();
        // do pass
        
        //----------------------------------------------------
        int pass_num = 18;//15;
        
        GBTConfig conf;
        conf.sampling_size_ratio = 0.5;
        conf.learning_rate = 0.1;//0.21;
        conf.tree_min_nodes = 22;
        conf.tree_depth = 3;
        conf.tree_number = 22;
        
        //----------------------------------------------------
        
        VC<VC<Entry>>passRes;
        for (int i = 0; i < pass_num; i++) {
            if (LOG_DEBUG) cerr << "Pass #" << i << endl;
            
            VC<Entry> rankList;
            rank(input_x, input_y, testing, conf, rankList);
            passRes.push_back(rankList);
        }
        
        double rankTime = getTime();
        
        // find mean
        VC<Entry>meanResults;
        for (int i = 0; i < Y; i++) {
            double meanDv = 0;
            int id = 0;
            for (int j = 0; j < pass_num; j++) {
                id = passRes[j][i].id;
                meanDv += passRes[j][i].dv;
            }
            meanDv /= pass_num;
            Entry mean_entry(id);
            mean_entry.dv = meanDv;
            meanResults.push_back(mean_entry);
        }
        
        double finishTime = getTime();
        
        //        VC<Entry>simRes;
        //        correctBySimilarity(training, meanResults, 0.0, simRes);
        
        if (LOG_DEBUG) cerr << "++++ OUT ++++" << endl;
        
        // sort to have highest rating at the top
        sort(meanResults.rbegin(), meanResults.rend());
        
        VI ids;
        for (int i = 0; i < Y; i++) {
            if (LOG_DEBUG) cerr << "ID: " << meanResults[i].id << ", activity: " << meanResults[i].dv << endl;
            
            ids.push_back(meanResults[i].id);
        }
        
        cerr << "pass_num: " << pass_num << ", learning_rate: " << conf.learning_rate << ", tree_min_nodes: " << conf.tree_min_nodes
        << ", tree_depth: " << conf.tree_depth <<  ", tree_number: " << conf.tree_number << endl;
        cerr << "Rank time: " << rankTime - startTime << ", full time: " << finishTime - startTime << endl;
        
        return ids;
    }
    
    void rank(const VC<VD> &input_x, const VD &input_y, const VC<Entry> &testing, const GBTConfig &conf, VC<Entry> &rank) {
        //        VC<VD> input_yy;
        //        int feature_dim = input_x[0].size();
        //        for (double y : input_y) {
        //            VD yList;
        //            for (int i = 0; i < feature_dim; i++) {
        //                yList.push_back(y);
        //            }
        //
        //            input_yy.push_back(yList);
        //        }
        
        // train
        GradientBoostingTree tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        PredictionForest *predictor = tree.fitGradientBoostingTree(input_x, input_y);
        //        ResultFunction *predictor = tree.learnGradientBoostingRanker(input_x, input_yy, .21);
        
        Printf("Features importance++++++++++++++++\n");
        VI importance = predictor->featureImportances();
        for (int i = 0; i < importance.size(); i++) {
            Printf("%i : %i | ", i, importance[i]);
        }
        Printf("\n");
        
        // predict
        int test_N = testing.size();
        for (int i = 0; i < test_N; i++) {
            Entry resEntry(testing[i].id);
            resEntry.dv = predictor->predict(testing[i].features);
            rank.push_back(resEntry);
        }
    }
    
    void correctBySimilarity(const VC<Entry> &training, const VC<Entry> &test, double simSplit, VC<Entry> &simResults) {
        // calculate MSE
        double mae = 0;
        VD vals;
        for (Entry e : test) {
            int testId = e.id;
            double testDv = e.dv;
            double dvSum = 0;
            double wSum = 0;
            
            double maxSim = -1;
            double maxVal = -1000;
            for (Entry trEntry : training) {
                double sim = matrix[testId][trEntry.id];
                if (trEntry.dv != Entry::NVL) {
                    dvSum += sim * trEntry.dv;
                    wSum += sim;
                    
                    if (sim > maxSim) {
                        maxSim = sim;
                        maxVal = trEntry.dv;
                    }
                }
            }
            
            double corr = dvSum / wSum;
            //            vals.push_back(corr);
            //            mae += abs(testDv - corr);
            
            vals.push_back(corr);
            mae += abs(testDv - maxVal);
            
            //            if (LOG_DEBUG) cerr << "TestDV: " << testDv << ", dvSum: " << dvSum << ", correction: " << corr << ", weights: " <<  wSum << endl;
            if (LOG_DEBUG) cerr << "TestDV: " << testDv << ", maxSim: " << maxSim << ", maxVal: " << maxVal << endl;
        }
        mae /= test.size();
        Printf("MAE: %f\n", mae);
        
        int index = 0;
        for (Entry e : test) {
            int testId = e.id;
            double testDv = e.dv;// * vals[index];
            
            //            double diff = testDv - vals[index];
            //            if (abs(diff) <= mae) {
            //                if (diff < 0) {
            //                    testDv =  vals[index] - mae / 2;
            //                } else if (diff > 0){
            //                    testDv =  vals[index] + mae / 2;
            //                }
            //
            //            }
            
            Printf("Entry: %i corrected with value: %f\n", testId, testDv);
            
            Entry resEntry(testId);
            resEntry.dv = testDv;
            simResults.push_back(resEntry);
            index++;
        }
    }
};

#ifndef FROM_TEST_RUNNER
/*
int main(int argc, const char * argv[]) {
    int X, Y;
    cin>>X;
    cin>>Y;
    
    Printf("X: %i, Y: %i\n",X ,Y);
    
    
    ActiveMolecules am;
    REP(i, X+Y) {
        VD similarities;
        REP(j, X+Y) {
            double sim;
            cin >> sim;
            similarities.push_back(sim);
        }
        // set similarities
        am.similarity(i, similarities);
    }
    
    Printf("Similarity set\n");
    
    VS trainingData;
    REP(i, X) {
        string t;
        cin>>t;
        trainingData.push_back(t);
    }
    VS testingData;
    REP(i, Y) {
        string t;
        cin>>t;
        testingData.push_back(t);
    }
    
    // do ranking
    VI ret = am.rank(trainingData, testingData);
    for (int r : ret) {
        cout<<r<<endl;
    }
    cout.flush();
    
    
    Printf("Training data size: %lu, testing data size: %lu, return size: %lu\n", trainingData.size(), testingData.size(), ret.size());
    
    return 0;
}
 */
#endif

inline void saveModel(string fn, VVD &model) {
    FILE *f = fopen(fn.c_str(), "a");
    for (VD &v : model) {
        for (double d : v) fprintf(f, "%.10lf ", d);
        fprintf(f, "\n");
    }
    fclose(f);
}