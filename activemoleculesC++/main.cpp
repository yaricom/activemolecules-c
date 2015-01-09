//
//  main.cpp
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/6/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//

#define LOCAL true

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
#define PB          push_back
#define S           size()
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


VS splt(string s, char c = ',') {
    VS all;
    int p = 0, np;
    while (np = s.find(c, p), np >= 0) {
        if (np != p)
            all.PB(s.substr(p, np - p));
        else
            all.PB("");
        p = np + 1;
    }
    if (p < s.size())
        all.PB(s.substr(p));
    return all;
}

static bool LOG_DEBUG = true;

double getTime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

struct Entry {
    static constexpr double NVL = -1000;
    
    // The number of features
    static constexpr int fCount = 22;
    
    int id;
    VD features;
    double dv;
    string formula;
    
    static double parse(const string &s) {
        return s == "" ? NVL : atof(s.c_str());
    }
    
    Entry(int sId) {
        id = sId;
    }
    
    Entry(int sId, const string &s) {
        id = sId;
        VS vs = splt(s, ',');
        REP(i, fCount) {
            if (i == 14) {
                formula = vs[i];
            } else {
                features.PB(parse(vs[i]));
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
    
    bool operator < (const Entry& entry) const {
        return (dv < entry.dv);
    }
};

struct GBTConfig {
    double sampling_size_ratio = 0.5;
    double learning_rate = 0.01;
    int tree_number = 130;
    int tree_min_nodes = 10;
    int tree_depth = 3;
};

void parseData(const VS &data, const int startIndex, VC<Entry> &v) {
    int index = startIndex;
    for (const string &s : data) {
        v.PB(Entry(index++, s));
    }
}

template<class bidiiter>
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
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
    RandomSample(int max, int number) {
        m_max = max;
        m_number = number;
    }
    
    VI get_sample_index() {
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
    Node(double value, int feature_index, double value_left, double value_right) {
        m_node_value = value;
        m_feature_index = feature_index;
        m_terminal_left = value_left;
        m_terminal_right = value_right;
    }
};

struct BestSplit {
    int m_feature_index;
    double m_node_value;
    bool m_status;
    
    // construction function
    BestSplit() {
        m_feature_index = 0;
        m_node_value = 0.0;
        m_status = false; // by default, it fails
    }
};

struct SplitRes {
    VC<VD> m_feature_left;
    VC<VD> m_feature_right;
    double m_left_value;
    double m_right_value;
    VD m_obs_left;
    VD m_obs_right;
    
    // construction function
    SplitRes() {
        m_left_value = 0.0;
        m_right_value = 0.0;
    }
};

struct ListData {
    double m_x;
    double m_y;
    
    ListData(double x, double y) {
        m_x = x;
        m_y = y;
    }
    
    bool operator < (const ListData& str) const {
        return (m_x < str.m_x);
    }
};

typedef enum _TerminalType {
    AVERAGE, MAXIMAL
}TerminalType;

class RegressionTree {
public:
    // class members
    int m_min_nodes;
    int m_max_depth;
    int m_current_depth;
    TerminalType m_type;
    
    // The root node
    Node *m_root = NULL;
    
    // construction function
    RegressionTree() {
        m_min_nodes = 10;
        m_max_depth = 3;
        m_current_depth = 0;
        m_type = AVERAGE;
    }
    
    // set parameters
    void setMinNodes(int min_nodes) {
        assert(min_nodes > 3);
        m_min_nodes = min_nodes;
    }
    
    void setDepth(int depth) {
        assert(depth > 0);
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
        int feature_num = feature_x.size();
        
        assert(feature_num == obs_y.size() && feature_num != 0);
        
        assert (m_min_nodes * 2 <= feature_num);
        
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
        
        int feature_num = feature_x.size();
        
        if (m_min_nodes * 2 > feature_num) {
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
            for (int loop_j = 0; loop_j < feature_num; loop_j++) {
                list_feature.PB(ListData(feature_x[loop_j][loop_i], obs_y[loop_j]));
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
            for (int loop_j = m_min_nodes; loop_j < feature_num; loop_j++) {
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
            for (int loop_j = m_min_nodes; loop_j <= feature_num - m_min_nodes - 1; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                double y = fetched_data.m_y;
                sum_left += y;
                count_left++;
                mean_left = sum_left / count_left;
                
                
                sum_right -= y;
                count_right--;
                mean_right = sum_right / count_right;
                
                
                current_err = -1 * count_left * mean_left * mean_left -
                count_right * mean_right * mean_right;
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
        
        int count = obs_y.size();
        for (int loop_i = 0; loop_i < count; loop_i++) {
            VD ith_feature = feature_x[loop_i];
            if (ith_feature[feature_index] < node_value) {
                // append to the left
                // feature
                split_res.m_feature_left.PB(ith_feature);
                // observation
                split_res.m_obs_left.PB(obs_y[loop_i]);
            } else {
                // append to the right
                split_res.m_feature_right.PB(ith_feature);
                split_res.m_obs_right.PB(obs_y[loop_i]);
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

class ResultFunction {
public:
    // class members
    double m_init_value;
    VC<RegressionTree> m_trees;
    double m_combine_weight;
    
    // construction function
    ResultFunction(double learning_rate) {
        m_init_value = 0.0;
        m_combine_weight = learning_rate;
    }
    
    
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
        
        for (const RegressionTree tree : m_trees) {
            re_res += m_combine_weight * tree.predict(feature_x);
        }
        
        return re_res;
    }
};


class GradientBoostingTree {
    // class members
    double m_sampling_size_ratio;
    double m_learning_rate;
    int m_tree_number;
    
    // tree related parameters
    int m_tree_min_nodes;
    int m_tree_depth;
    
public:

    GradientBoostingTree(double sample_size_ratio, double learning_rate,
                                int tree_number, int tree_min_nodes, int tree_depth) {
        // This will be called when initialize the class with parameters
        
        /*
         *  Check the validity of numbers
         */
        assert(sample_size_ratio > 0 && learning_rate > 0 && tree_number > 0 && tree_min_nodes >= 3 && tree_depth > 0);
        
        // In the gradient method, the portion of "sample_size_ration"
        // will be sampled without
        // replacement.
        m_sampling_size_ratio = sample_size_ratio;
        
        // Set learning rate or the shrink-age factor
        m_learning_rate = learning_rate;
        
        // set the number of trees
        m_tree_number = tree_number;
        
        // set tree parameters
        m_tree_min_nodes = tree_min_nodes;
        m_tree_depth = tree_depth;
    }
    
    /**
     * Fits a regression function using the Gradient Boosting Tree method.
     * On success, return function; otherwise, return null.
     *
     * @param input_x the input features
     * @param input_y the ground truth values - one per features row
     */
    ResultFunction fitGradientBoostingTree(const VC<VD> &input_x, const VD &input_y) {
        
        // initialize the final result
        ResultFunction res_fun(m_learning_rate);
        
        // get the feature dimension
        int feature_num = input_y.size();
        
        assert(feature_num == input_x.size() && feature_num > 0);
        
        // get an initial guess of the function
        double mean_y = 0.0;
        for (double d : input_y) {
            mean_y += d;
        }
        mean_y = mean_y / feature_num;
        res_fun.m_init_value = mean_y;
        
        
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
                gradient.PB(d - h_value[index]);
                
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
                    train_y.PB(gradient[sel_index]);
                    train_x.PB(input_x[sel_index]);
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
                
                res_fun.m_trees.PB(tree);
                
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
                res_fun.m_trees.PB(tree);
                
                // update h_value information, prepare for the next iteration
                for (int loop_index = 0; loop_index < feature_num; loop_index++) {
                    h_value[loop_index] += m_learning_rate *
                    tree.predict(input_x[loop_index]);
                }
            }
            
            // next iteration
            iter_index++;
        }
        
        // set the learning rate and return
        // res_fun.m_combine_weight = m_learning_rate;
        
        return res_fun;
    }
};

class ActiveMolecules {
    int X;
    int Y;
    int M;
    VC<VD>matrix;
    
    public:
    
    int similarity(int moleculeID, VD &similarities) {
        if (!matrix.S) {
            M = matrix.S;
        }
        VD row;
        for (double &d : similarities) {
            row.PB(d);
        }
        matrix.PB(row);
        
        return 0;
    }
    
    VI rank(const VS &train, const VS &test) {
        X = train.size();
        Y = test.size();
        
        cerr << "GB SC Training length:" << X << ", testing length: " << Y << endl;
        
        // parse data
        VC<Entry> training;
        VC<Entry> testing;
        parseData(train, 0, training);
        parseData(test, X, testing);
        
        assert(training.size() == X && testing.size() == Y);
        
        // prepare data
        VC<VD> input_x;
        VD input_y;
        for (Entry e : training) {
            if (e.dv > Entry::NVL) {
                // check for DV set only for training data and add TEST data without checks
                input_x.PB(e.features);
                input_y.PB(e.dv);
            } else if (LOG_DEBUG){
//                cerr << "Entry id: " << e.id << " missing DV" << endl;
            }
        }
        
        if (LOG_DEBUG) cerr << "Real train size: " << input_x.size() << endl;
        
        double startTime = getTime();
        // do pass
        
        //----------------------------------------------------
        int pass_num = 1;//9;
        
        GBTConfig conf;
        conf.sampling_size_ratio = 0.5;
        conf.learning_rate = 0.01;//0.19;//0.21;
        conf.tree_min_nodes = 10;//22;
        conf.tree_depth = 3;//1;
        conf.tree_number = 100;//22;
        
        double simSplit = 0;//0.8;
        //----------------------------------------------------
        
        VC<VC<Entry>>passRes;
        for (int i = 0; i < pass_num; i++) {
            if (LOG_DEBUG) cerr << "Pass #" << i << endl;
            
            VC<Entry> rankList;
            rank(input_x, input_y, testing, conf, rankList);
            passRes.PB(rankList);
        }

        double rankTime = getTime();
        
        // find mean
        VC<Entry>meanResults;
        for (int i = 0; i < Y; i++) {
            double meanDv = 0;
            int id = 0;
            for (int j = 0; j < pass_num; j++) {
                Entry res_entry = passRes[j][i];
                id = res_entry.id;
                meanDv += res_entry.dv;
            }
            meanDv /= pass_num;
            Entry mean_entry(id);
            mean_entry.dv = meanDv;
            meanResults.PB(mean_entry);
        }
        
        // correct by similarity
        VC<Entry> simResults;
        correctBySimilarity(training, meanResults, simSplit, simResults);
        
        double finishTime = getTime();
        
        if (LOG_DEBUG) cerr << "++++ OUT ++++" << endl;
        
        // sort to have highest rating at the top
        sort(simResults.rbegin(), simResults.rend());
        
        VI ids;
        for (Entry e : simResults) {
//            if (LOG_DEBUG) cerr << "ID: " << e.id << ", activity: " << e.dv << endl;
            
            ids.PB(e.id);
        }
        
        cerr << "pass_num: " << pass_num << ", learning_rate: " << conf.learning_rate << ", tree_number: " << conf.tree_number << ", split: " << simSplit << endl;
        cerr << "Rank time: " << rankTime - startTime << ", full time: " << finishTime - startTime << endl;
        
        return ids;
    }
    
private:
    
    void correctBySimilarity(const VC<Entry> &training, const VC<Entry> &test, double simSplit, VC<Entry> &simResults) {
        // calculate MSE
        double mae = 0;
        double mse = 0;
        double yMax = -10000;
        double yMin = 10000;
        VD vals;
        VD dvRel;
        int corrCount = 0;
        for (Entry e : test) {
            int testId = e.id;
            double testDv = e.dv;
            double dvSum = 0;
            double wSum = 0;

            for (Entry trEntry : training) {
                double sim = matrix[testId][trEntry.id];
                if (trEntry.dv > Entry::NVL && sim > simSplit) {
                    dvSum += sim * trEntry.dv;
                    wSum += sim;
                }
            }
            
            double corr = 0;
            if (wSum > 0) {
                corr = dvSum / training.size();// wSum;
                double ae = abs(testDv - corr);
                mae += ae;
                mse += ae * ae;

                vals.PB(corr);
                
                double err = ae / testDv;
                dvRel.PB(err);
                
                corrCount++;
            } else {
                dvRel.PB(0);
            }
            
            // find max/min
            if (testDv > yMax) {
                yMax = testDv;
            }
            if (testDv < yMin) {
                yMin = testDv;
            }

            if (LOG_DEBUG) cerr << "Entry: " << testId <<  " testDV: " << testDv <<  ", correction: " << corr << ", dvSum: " << dvSum << ", weights: " <<  wSum << endl;
        }
        
        // -1 to compensate last iteration
        mae /= (corrCount - 1);
        mse /= (corrCount - 1);
        // root-mean-square deviation
        double rmse = sqrt(mse);
        // normalized root-mean-square deviation
        double nrmse = rmse / (yMax - yMin);
        
        if (LOG_DEBUG) cerr << corrCount << " to be corrected" << endl;
        if (LOG_DEBUG) cerr << "MAE: " << mae << ", MSE: " << mse << ", RMSE: " << rmse << ", NRSME: " << nrmse << ", Ymax/Ymin: " << yMax << "/" << yMin << endl;
        
        int index = 0;
        corrCount = 0;
        for (Entry e : test) {
            int testId = e.id;
            double testDv = e.dv;
            double correction = vals[index];
            double dvCoef = dvRel[index];
            
            if (correction != 0) {
                
//                double diff = testDv - correction;
//                if (abs(diff) > nrmse) {
//                    testDv = correction;
//                
//                    corrCount++;
//                    
//                    if (LOG_DEBUG) cerr << "Entry: " << testId << " corrected with value: " << testDv << ", diff: " << diff << endl;
//                }
            
//                if (dvCoef < mae) {
                    testDv -= correction;
                    
                    corrCount++;
                    
                    if (LOG_DEBUG) cerr << "Entry: " << testId << " corrected with value: " << testDv << ", dvCoef: " << dvCoef << endl;
//                }
            }
            
            
            Entry resEntry(testId);
            resEntry.dv = testDv;
            simResults.PB(resEntry);
            index++;
        }
        
        if (LOG_DEBUG) cerr << corrCount << " was corrected" << endl;
    }
    
    void rank(const VC<VD> &input_x, const VD &input_y, const VC<Entry> &testing, const GBTConfig &conf, VC<Entry> &rank) {
        // train
        GradientBoostingTree tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        ResultFunction predictor = tree.fitGradientBoostingTree(input_x, input_y);
        
        // predict
        int test_N = testing.size();
        for (int i = 0; i < test_N; i++) {
            Entry testEntry = testing[i];
            Entry resEntry(testEntry.id);
            resEntry.dv = predictor.predict(testEntry.features);
            rank.PB(resEntry);
        }
    }
};

int main(int argc, const char * argv[]) {
    int X, Y;
    cin>>X;
    cin>>Y;
    
    cerr << "X: " << X << ", Y: " << Y << endl;
    
    
    ActiveMolecules am;
    REP(i, X+Y) {
        VD similarities;
        REP(j, X+Y) {
            double sim;
            cin >> sim;
            similarities.PB(sim);
        }
        // set similarities
        am.similarity(i, similarities);
    }
    
    cerr << "Similarity set" << endl;
    
    VS trainingData;
    REP(i, X) {
        string t;
        cin>>t;
        trainingData.PB(t);
    }
    VS testingData;
    REP(i, Y) {
        string t;
        cin>>t;
        testingData.PB(t);
    }
    
    // do ranking
    VI ret = am.rank(trainingData, testingData);
    for (int r : ret) {
        cout<<r<<endl;
    }
    cout.flush();
    
    
    cerr << "Training data size: " << trainingData.S << ", testing data size: " << testingData.S << ", return size: " << ret.S;
    
    return 0;
}
