//
//  WeightedKNN.h
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/12/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//

#ifndef activemoleculesC___WeightedKNN_h
#define activemoleculesC___WeightedKNN_h

//===============================
//xx is the total number of attributes in the data set INCLUDING the class attribute
//#define NO_OF_ATT xx
//yy is the number of classes
//#define NO_OF_CLASSES yy
//==============================

//
// The active molecules data
//
//xx is the total number of attributes in the data set INCLUDING the class attribute
#define NO_OF_ATT 22
//yy is the number of classes
#define NO_OF_CLASSES 51

//The Heart Data Set
//#define NO_OF_ATT 14
//#define NO_OF_CLASSES 2

//The Hill Valley Data Set
//#define NO_OF_ATT 101
//#define NO_OF_CLASSES 2

//Wine Data Set
//#define NO_OF_ATT 14
//#define NO_OF_CLASSES 3

//Heart-1 Data Set
//#define NO_OF_ATT 14
//#define NO_OF_CLASSES 5
/*-------------------------------------------------*/

/* The number of nearest neighbors to use. */
#define K 3
#define LEARNING_RATE 0.005

#define uint	unsigned int

enum MODE
{
    TRAINING = 0,
    TESTING,
    VALIDATING
};

//Backward Elimination
extern bool isBEAttIncluded[];

//Attribute Weighting
extern double attWeights[];

class TrainingExample
{
public:
    // Unique Index
    uint index;
    // Values of all Attributes for an instance
    double Value [NO_OF_ATT];
    // Euclidean Distance
    double Distance;
    // Instance Weight
    double Weight;
    // Is the instance near to anyone
    bool isNearest2AtleastSome;
    
    TrainingExample()
    {
        for(int i = 0; i < NO_OF_ATT; i++)
            Value[i] = 0.0;
        Distance = 0.0;
        Weight = 0.0;
        index = 0;
        isNearest2AtleastSome = false;
    }
    
    TrainingExample(double *a)
    {
        for(int i = 0; i < NO_OF_ATT; i++)
            Value[i] = a[i];
        Distance = 0.0;
        Weight = 0.0;
        index = 0;
        isNearest2AtleastSome = false;
    }
    
    ~TrainingExample()
    {
        for(int i = 0; i < NO_OF_ATT; i++)
            Value[i] = 0.0;
        Distance = 0.0;
        Weight = 0.0;
        index = 0;
    }
    
    void SetVal(double *a)
    {
        for(int i = 0; i < NO_OF_ATT; i++)
            Value[i] = a[i];
    }
    
    void GetVal(double *a)
    {
        for(int i = 0; i < NO_OF_ATT; i++)
            a[i] = Value[i];
    }
    
    //Not using this normalization anymore
    //Using Standard Deviation instead
    void NormalizeVals ()
    {
        for (int i = 0; i < NO_OF_ATT - 1; i++)
        {
            Value[i] = Value[i] / (1.0 + Value[i]);
        }
    }
    
};

/* Using a list to store the training and testing examples. */
typedef std::list<TrainingExample, std::allocator<TrainingExample> > TRAINING_EXAMPLES_LIST;

/* Prints example to stderr */
void printExample(TrainingExample &example);

/* Function Definitions */
/////////////////////////////KNN Algorithms//////////////////////////
/* K Nearest Neighbor Algorithm (All attributes treated equally) */
float SimpleKNN (TRAINING_EXAMPLES_LIST *trainList,
                 int trainExamples,
                 TRAINING_EXAMPLES_LIST *testList,
                 int testExamples);

/* Attribute Weighted K Nearest Neighbor Algorithm */
float AttributeWKNN (TRAINING_EXAMPLES_LIST *trainList,
                     int trainExamples,
                     TRAINING_EXAMPLES_LIST *testList,
                     int testExamples);

/* Instance Weighted K Nearest Neighbor Algorithm */
float InstanceWKNN (TRAINING_EXAMPLES_LIST *trainList,
                    int trainExamples,
                    TRAINING_EXAMPLES_LIST *testList,
                    int testExamples);

/* K Nearest Neighbor Algorithm with Backward Elimination */
float BackwardElimination (TRAINING_EXAMPLES_LIST *trainList,
                           int trainExamples,
                           TRAINING_EXAMPLES_LIST *testList,
                           int testExamples);

/////////////////////////////KNN Algorithm Classifiers//////////////////////////
/**
 * Do classification by Attribute Weighted K Nearest Neighbor Algorithm
 * and returns list of classes in the same order as in test list examples
 *
 * trainList - List of training examples
 * testList - List of testing examples
 */
std::vector<int>classifyBySimpleAttributeWKNN (TRAINING_EXAMPLES_LIST *trainList,
                                               TRAINING_EXAMPLES_LIST *testList);

/**
 * Do classification by K Nearest Neighbor Algorithm (All attributes treated equally)
 * and returns list of classes in the same order as in test list examples
 *
 * trainList - List of training examples
 * testList - List of testing examples
 */
std::vector<int>classifyBySimpleKNN (TRAINING_EXAMPLES_LIST *trainList,
                                     TRAINING_EXAMPLES_LIST *testList);
/**
 * Do classification by Instance Weighted K Nearest Neighbor Algorithm
 * and returns list of classes in the same order as in test list examples
 *
 * trainList - List of training examples
 * testList - List of testing examples
 */
std::vector<int>classifyByInstanceWKNN (TRAINING_EXAMPLES_LIST *trainList,
                                        TRAINING_EXAMPLES_LIST *testList);

/**
 * Do classification by K Nearest Neighbor Algorithm with Backward Elimination 
 * and returns list of classes in the same order as in test list examples
 */
std::vector<int>classifyByKNNBackwardElimination (TRAINING_EXAMPLES_LIST *trainList,
                                                  TRAINING_EXAMPLES_LIST *testList);
///////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////////Learning Functions for Instance and Attribute Weighted KNN////////
/* Learning weights by running KNN on training data. */
float LearnWeights (TRAINING_EXAMPLES_LIST *tlist,
                    TRAINING_EXAMPLES_LIST data,
                    int iterations, int numExamples,
                    MODE mode, int desiredAccuracy,
                    bool isAttWeightedKNN);

/* Adjust weights by using Gradient Descent */
void AdjustWeightsByGradientDescent (double *qvalue,
                                     TRAINING_EXAMPLES_LIST *tlist,
                                     double error, uint *index,
                                     bool isAttWeightedKNN);

/* Learn weights by cross validation */
float CrossValidate(TRAINING_EXAMPLES_LIST *data, int iterations,
                    int numExamples, bool isAttWKNN);
//////////////////////////////////////////////////////////////////////////

/* Finds K nearest neighbors and predicts class according to algorithm used. */
int PredictByKNN (TRAINING_EXAMPLES_LIST *tlist, double *query,
                  bool isWeightedKNN, uint *index, MODE mode,
                  bool isBE, bool isAttWeightedKNN);

/* Test KNN algorithms */
int TestKNN (TRAINING_EXAMPLES_LIST *tlist, TRAINING_EXAMPLES_LIST data,
             bool isWeighted, MODE mode, bool isBackwardElimination,
             bool isAttWKNN);


/* Reads the training and testing data into the list. */
bool readData4File (char *filename,
                    TRAINING_EXAMPLES_LIST *rlist,
                    int *rlistExamples);

/* Comparison function used during sorting data. */
bool compare(const TrainingExample t1, const TrainingExample t2);

/* Normalizes data values using standard deviation. */
void NormalizeByStandardDeviation (TRAINING_EXAMPLES_LIST *trainList, int trainExamples);

#endif
