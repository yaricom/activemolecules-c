#define LOCAL true

#ifdef LOCAL
#include "stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include "WeightedKNN.h"


//Backward Elimination
bool isBEAttIncluded[NO_OF_ATT - 1];

//Attribute Weighting
double attWeights[NO_OF_ATT - 1];

using namespace std;

void printExample(TrainingExample &example) {
    cerr << "TrainingExample id: " << example.index << ", weight: " << example.Weight << ", values [";
    for(int i = 0; i < NO_OF_ATT; i++){cerr << example.Value[i] << ", ";}
    cerr << "]" << endl;
}

void printExamples(TRAINING_EXAMPLES_LIST &data) {
    TRAINING_EXAMPLES_LIST::iterator testIter;
    TrainingExample tmpTestObj;
    
    for (testIter = data.begin(); testIter != data.end(); ++testIter) {
        tmpTestObj = *testIter;
        printExample(tmpTestObj);
    }
}

void BackwardEliminationInit ();
void InitAttWeights ();

/*-----------------------------------------------------------------*/
/* trainList - List of training examples                           */
/* trainExamples - # of training examples                          */
/* testList - List of testing examples                             */
/* testExamples - # of testing examples                            */
/*-----------------------------------------------------------------*/
float SimpleKNN (TRAINING_EXAMPLES_LIST *trainList, int trainExamples,
                 TRAINING_EXAMPLES_LIST *testList, int testExamples)
{
    bool isInstanceWKNN = false;
    MODE mode			= TESTING;
    bool isBackwardElim = false;
    bool isAttrWKNN		= false;
    
    cerr<<endl<<"Testing Simple KNN(Without Weights)."<<endl;
    
    /* Test Simple KNN */
    int CCI = TestKNN(trainList, *testList, isInstanceWKNN, mode,
                      isBackwardElim, isAttrWKNN);
    
    float accuracy = (float)(((float)CCI/(float)testExamples)*100);
    
    cerr<<"----------------------KNN----------------------"<<endl;
    cerr<<"Number of Training Examples      # "<<trainExamples<<endl;
    cerr<<"Number of Testing Examples       # "<<testExamples<<endl;
    cerr<<"K used                           = "<<K<<endl;
    cerr<<"Correctly Classified Instances   # "<<CCI<<endl;
    cerr<<"Incorrectly Classified Instances # "<<testExamples - CCI<<endl;
    cerr<<"Accuracy (%)                     = "<<accuracy<<endl;
    cerr<<"-----------------------------------------------"<<endl<<endl;
    
    return accuracy;
}

/*-----------------------------------------------------------------*/
/* trainList - List of training examples                           */
/* trainExamples - # of training examples                          */
/* testList - List of testing examples                             */
/* testExamples - # of testing examples                            */
/*-----------------------------------------------------------------*/
float InstanceWKNN (TRAINING_EXAMPLES_LIST *trainList, int trainExamples,
                    TRAINING_EXAMPLES_LIST *testList, int testExamples)
{
    bool isInstanceWKNN		= true;
    bool isBackwardElim		= false;
    bool isAttrWKNN			= false;
    int no_of_iterations	= 25;
    int desiredAccuracy		= 50;//85;
    
    cerr<<endl<<"Starting Instance Weighted KNN..."<<endl;
    
    /* Learn weights by cross validation (3 fold) on training set */
    float accuracy = CrossValidate (trainList, no_of_iterations,
                                    trainExamples, isAttrWKNN);
    /* Learn weights on the training set */
    LearnWeights (trainList, *trainList, no_of_iterations, trainExamples,
                  TRAINING, desiredAccuracy, isAttrWKNN);
    
    /* Test the trained weights on test set */
    int CCI = TestKNN (trainList, *testList, isInstanceWKNN,
                       TESTING, isBackwardElim, isAttrWKNN);
    
    accuracy = (float)(((float)CCI/(float)testExamples)*100);
    
    cerr<<"-------Instance Weighted-KNN----------------------------"<<endl;
    cerr<<"Number of Training Examples      # "<<trainExamples<<endl;
    cerr<<"Number of Testing Examples       # "<<testExamples<<endl;
    cerr<<"K used                           = "<<K<<endl;
    cerr<<"Correctly Classified Instances   # "<<CCI<<endl;
    cerr<<"Incorrectly Classified Instances # "<<testExamples - CCI<<endl;
    cerr<<"Accuracy (%)                     = "<<accuracy<<endl;
    cerr<<"--------------------------------------------------------"<<endl;
    
    return accuracy;
}

/*-----------------------------------------------------------------*/
/* trainList - List of training examples                           */
/* trainExamples - # of training examples                          */
/* testList - List of testing examples                             */
/* testExamples - # of testing examples                            */
/*-----------------------------------------------------------------*/
float BackwardElimination (TRAINING_EXAMPLES_LIST *trainList,
                           int trainExamples,
                           TRAINING_EXAMPLES_LIST *testList,
                           int testExamples)
{
    float accuracy = 0.0f;
    int CCI = 0;
    int noOfAttDeleted = 0;
    
    bool isInstanceWKNN		= false;
    MODE mode				= TESTING;
    bool isBackwardElim		= true;
    bool isAttrWKNN			= false;
    
    cerr<<endl<<"Starting Backward Elimination..."<<endl;
    cerr<<"--------------------Backward Elimination--------------------"<<endl;
    
    /* Initially all the attributes will be included in KNN */
    BackwardEliminationInit ();
    
    /* Test KNN with all the attributes */
    CCI = TestKNN (trainList, *trainList, isInstanceWKNN,
                   mode, isBackwardElim, isAttrWKNN);
    accuracy = (float)(((float)CCI/(float)trainExamples)*100);
    
    cerr << "Initial Accuracy on training data with " << NO_OF_ATT - 1
    << " attributes = " << accuracy << "%" <<endl;
    
    for (int i = 0; i < NO_OF_ATT - 1; i++) {
        /* Delete one attribute at a time.                      */
        /* If the accuracy has decreased restore the attribute  */
        /* else let the attribute remain deleted.               */
        isBEAttIncluded[i] = false;
        CCI = TestKNN (trainList, *trainList, isInstanceWKNN,
                       mode, isBackwardElim, isAttrWKNN);
        float tmpAcc = (float)(((float)CCI/(float)trainExamples) * 100);
        if(tmpAcc >= accuracy) {
            accuracy = tmpAcc;
            noOfAttDeleted++;
            cerr << "Attribute " << i << " irrelevant and removed!" <<endl;
        } else {
            isBEAttIncluded[i] = true;
        }
    }
    
    cerr << "Backward Elimination achieves " << accuracy << "% accuracy with "
    << NO_OF_ATT - 1 - noOfAttDeleted << " attributes \n on training data." <<endl;
    
    
    /* Test KNN again with eliminated attributes on the test data. */
    CCI = TestKNN (trainList, *testList, isInstanceWKNN, mode, isBackwardElim, isAttrWKNN);
    accuracy = (float)(((float)CCI/(float)testExamples)*100);
    
    cerr<<"Number of Testing Examples       # "<<testExamples<<endl;
    cerr<<"K used                           = "<<K<<endl;
    cerr<<"Correctly Classified Instances   # "<<CCI<<endl;
    cerr<<"Incorrectly Classified Instances # "<<testExamples - CCI<<endl;
    cerr<<"Backward Elimination achieves " <<accuracy << "% accuracy with "
    << NO_OF_ATT - 1 - noOfAttDeleted << " attributes \n on testing data." <<endl;
    cerr<<"------------------------------------------------------------"<<endl;
    
    return accuracy;
}

/////////////////////////////KNN Algorithm Classifiers//////////////////////////
/**
 * Do classification by Attribute Weighted K Nearest Neighbor Algorithm
 * and returns list of classes in the same order as in test list examples
 *
 * trainList - List of training examples
 * testList - List of testing examples
 */
std::vector<int>classifyBySimpleAttributeWKNN (TRAINING_EXAMPLES_LIST *trainList,
                                               TRAINING_EXAMPLES_LIST *testList) {
    int trainExamples = trainList->size();
    int testExamples = testList->size();
    
    int no_of_iterations	= 25;
    int desiredAccuracy		= 85;
    bool isAttrWKNN			= true;
    bool isInstanceWKNN		= false;
    bool isBackwardElim		= false;
    
    cerr<<endl<< "Starting Attribute Weighted KNN classifier" <<endl;
    cerr<<"------------------------------------------------------------"<<endl;
    cerr << "Training data size: " << trainExamples << ", test data size: " << testExamples << endl;
    
    //
    // Do train
    //
    
    /* Every attribute is associated with a weight. */
    /* Initialize the weights with random values.   */
    InitAttWeights ();
    
    /* Learn weights by cross validation (3 fold) on training set */
    float accuracy = CrossValidate (trainList, no_of_iterations,
                                    trainExamples, isAttrWKNN);
    
    cerr << "CrossValidate accuracy: " << accuracy << endl;
    
    /* Learn weights on the whole training set */
    no_of_iterations = 100;
    LearnWeights (trainList, *trainList, no_of_iterations, trainExamples,
                  TRAINING, desiredAccuracy, isAttrWKNN);
    
    //
    // Do test
    //
    
    MODE mode = TESTING;
    TRAINING_EXAMPLES_LIST::iterator testIter;
    TrainingExample tmpTestObj;
    uint index[K];
    std::vector<int>classes;
    
    for (testIter = testList->begin(); testIter != testList->end(); ++testIter) {
        tmpTestObj = *testIter;
        /* Predict the class for the query point */
        int predictedClass = PredictByKNN(trainList, tmpTestObj.Value, isInstanceWKNN,
                                          index, mode, isBackwardElim, isAttrWKNN);
        
        cerr << "Entry id: " << tmpTestObj.index << ", predicted class: " << predictedClass <<endl;
        
        classes.push_back(predictedClass);
    }

    
    cerr<<"-------Attribute Weighted-KNN---------------------------"<<endl;
    cerr<<"Number of Training Examples      # "<<trainExamples<<endl;
    cerr<<"Number of Testing Examples       # "<<testExamples<<endl;
    cerr<<"K used                           = "<<K<<endl;
    cerr<<"Classified classes Instances     # " << classes.size() <<endl;
    cerr<<"--------------------------------------------------------"<<endl;
    
    return classes;
}

/**
 * Do classification by K Nearest Neighbor Algorithm (All attributes treated equally)
 * and returns list of classes in the same order as in test list examples
 *
 * trainList - List of training examples
 * testList - List of testing examples
 */
std::vector<int>classifyBySimpleKNN (TRAINING_EXAMPLES_LIST *trainList,
                                     TRAINING_EXAMPLES_LIST *testList) {
    int trainExamples = trainList->size();
    int testExamples = testList->size();
    
    bool isInstanceWKNN = false;
    MODE mode			= TESTING;
    bool isBackwardElim = false;
    bool isAttrWKNN		= false;
    
    cerr<<endl<<"Simple KNN(Without Weights) classification"<<endl;
    cerr<<"------------------------------------------------------------"<<endl;
    cerr << "Training data size: " << trainExamples << ", test data size: " << testExamples << endl;
    
    TRAINING_EXAMPLES_LIST::iterator testIter;
    TrainingExample tmpTestObj;
    uint index[K];
    std::vector<int>classes;
    
    for (testIter = testList->begin(); testIter != testList->end(); ++testIter) {
        tmpTestObj = *testIter;
        /* Predict the class for the query point */
        int predictedClass = PredictByKNN(trainList, tmpTestObj.Value, isInstanceWKNN,
                                          index, mode, isBackwardElim, isAttrWKNN);
        
        cerr << "Entry id: " << tmpTestObj.index << ", predicted class: " << predictedClass <<endl;
        
        classes.push_back(predictedClass);
    }
    
    cerr<<"----------------------KNN----------------------"<<endl;
    cerr<<"Number of Training Examples      # "<<trainExamples<<endl;
    cerr<<"Number of Testing Examples       # "<<testExamples<<endl;
    cerr<<"K used                           = "<<K<<endl;
    cerr<<"Classified classes Instances     # " << classes.size() <<endl;
    cerr<<"-----------------------------------------------"<<endl<<endl;
    
    return classes;
    
}

/**
 * Do classification by Instance Weighted K Nearest Neighbor Algorithm
 * and returns list of classes in the same order as in test list examples
 *
 * trainList - List of training examples
 * testList - List of testing examples
 */
std::vector<int>classifyByInstanceWKNN (TRAINING_EXAMPLES_LIST *trainList,
                                        TRAINING_EXAMPLES_LIST *testList) {
    int trainExamples = trainList->size();
    int testExamples = testList->size();
    bool isInstanceWKNN		= true;
    bool isBackwardElim		= false;
    bool isAttrWKNN			= false;
    int no_of_iterations	= 25;
    int desiredAccuracy		= 50;//85;
    
    cerr<<endl<<"Starting Instance Weighted KNN classification"<<endl;
    cerr<<"------------------------------------------------------------"<<endl;
    cerr << "Training data size: " << trainExamples << ", test data size: " << testExamples << endl;
    
    //
    // Do train
    //
    
    /* Learn weights by cross validation (3 fold) on training set */
    float accuracy = CrossValidate (trainList, no_of_iterations,
                                    trainExamples, isAttrWKNN);
    /* Learn weights on the training set */
    LearnWeights (trainList, *trainList, no_of_iterations, trainExamples,
                  TRAINING, desiredAccuracy, isAttrWKNN);
    cerr << "Achieved cross validation accuracy: " << accuracy << endl;
    
    
    //
    // Do test
    //
    
    MODE mode = TESTING;
    TRAINING_EXAMPLES_LIST::iterator testIter;
    TrainingExample tmpTestObj;
    uint index[K];
    std::vector<int>classes;
    
    for (testIter = testList->begin(); testIter != testList->end(); ++testIter) {
        tmpTestObj = *testIter;
        /* Predict the class for the query point */
        int predictedClass = PredictByKNN(trainList, tmpTestObj.Value, isInstanceWKNN,
                                          index, mode, isBackwardElim, isAttrWKNN);
        
        cerr << "Entry id: " << tmpTestObj.index << ", predicted class: " << predictedClass <<endl;
        
        classes.push_back(predictedClass);
    }
    
    cerr<<"-------Instance Weighted-KNN----------------------------"<<endl;
    cerr<<"Number of Training Examples      # "<<trainExamples<<endl;
    cerr<<"Number of Testing Examples       # "<<testExamples<<endl;
    cerr<<"K used                           = "<<K<<endl;
    cerr<<"Classified classes Instances     # " << classes.size() <<endl;
    cerr<<"--------------------------------------------------------"<<endl;
    
    return classes;
}

/**
 * Do classification by K Nearest Neighbor Algorithm with Backward Elimination
 * and returns list of classes in the same order as in test list examples
 *
 * trainList - List of training examples
 * testList - List of testing examples
 */
std::vector<int>classifyByKNNBackwardElimination (TRAINING_EXAMPLES_LIST *trainList,
                                                  TRAINING_EXAMPLES_LIST *testList) {
    int trainExamples = trainList->size();
    int testExamples = testList->size();
    
    float accuracy = 0.0f;
    int CCI = 0;
    int noOfAttDeleted = 0;
    
    bool isInstanceWKNN		= false;
    MODE mode				= TESTING;
    bool isBackwardElim		= true;
    bool isAttrWKNN			= false;
    
    cerr << endl << "Starting Backward Elimination Classification" << endl;
    cerr<<"------------------------------------------------------------"<<endl;
    cerr << "Training data size: " << trainExamples << ", test data size: " << testExamples << endl;
    
    //
    // Do training
    //
    
    /* Initially all the attributes will be included in KNN */
    BackwardEliminationInit ();
    
    /* Test KNN with all the attributes */
    CCI = TestKNN (trainList, *trainList, isInstanceWKNN,
                   mode, isBackwardElim, isAttrWKNN);
    accuracy = (float)(((float)CCI/(float)trainExamples)*100);
    
    cerr << "Initial Accuracy on training data with " << NO_OF_ATT - 1
    << " attributes = " << accuracy << "%" <<endl;
    
    for (int i = 0; i < NO_OF_ATT - 1; i++) {
        /* Delete one attribute at a time.                      */
        /* If the accuracy has decreased restore the attribute  */
        /* else let the attribute remain deleted.               */
        isBEAttIncluded[i] = false;
        CCI = TestKNN (trainList, *trainList, isInstanceWKNN,
                       mode, isBackwardElim, isAttrWKNN);
        float tmpAcc = (float)(((float)CCI/(float)trainExamples) * 100);
        if(tmpAcc >= accuracy) {
            accuracy = tmpAcc;
            noOfAttDeleted++;
            cerr << "Attribute " << i << " is an irrelevant and was removed!" <<endl;
        } else {
            isBEAttIncluded[i] = true;
        }
    }
    
    cerr << "Backward Elimination achieves " << accuracy << "% accuracy with "
    << NO_OF_ATT - 1 - noOfAttDeleted << " attributes on training data." <<endl;
    
    
    //
    // Do prediction
    //
    TRAINING_EXAMPLES_LIST::iterator testIter;
    TrainingExample tmpTestObj;
    uint index[K];
    std::vector<int>classes;
    
    for (testIter = testList->begin(); testIter != testList->end(); ++testIter) {
        tmpTestObj = *testIter;
        /* Predict the class for the query point */
        int predictedClass = PredictByKNN(trainList, tmpTestObj.Value, isInstanceWKNN,
                                          index, mode, isBackwardElim, isAttrWKNN);
        
        cerr << "Entry id: " << tmpTestObj.index << ", predicted class: " << predictedClass <<endl;
        
        classes.push_back(predictedClass);
    }
    
    cerr<<"Number of Testing Examples       # " << testExamples <<endl;
    cerr<<"K used                           = " << K <<endl;
    cerr<<"Classified classes Instances     # " << classes.size() <<endl;
    cerr<<"------------------------------------------------------------"<<endl;
    
    return classes;
}

/*-----------------------------------------------------------------*/
/* trainList - List of training examples                           */
/* trainExamples - # of training examples                          */
/* testList - List of testing examples                             */
/* testExamples - # of testing examples                            */
/*-----------------------------------------------------------------*/
float AttributeWKNN (TRAINING_EXAMPLES_LIST *trainList, int trainExamples,
                     TRAINING_EXAMPLES_LIST *testList, int testExamples)
{
    int no_of_iterations	= 25;
    int desiredAccuracy		= 85;
    bool isAttrWKNN			= true;
    bool isInstanceWKNN		= false;
    bool isBackwardElim		= false;
    
    cerr<<endl<<"Starting Attribute Weighted KNN..."<<endl;
    
    /* Every attribute is associated with a weight. */
    /* Initialize the weights with random values.   */
    InitAttWeights ();
    
    /* Learn weights by cross validation (3 fold) on training set */
    float accuracy = CrossValidate (trainList, no_of_iterations,
                                    trainExamples, isAttrWKNN);
    
    cerr << "CrossValidate accuracy: " << accuracy << endl;
    
    /* Learn weights on the whole training set */
    no_of_iterations = 100;
    LearnWeights (trainList, *trainList, no_of_iterations, trainExamples,
                  TRAINING, desiredAccuracy, isAttrWKNN);
    
    /* Test the trained weights with the test set. */
    int CCI = TestKNN (trainList, *testList, isInstanceWKNN, TESTING,
                       isBackwardElim, isAttrWKNN);
    accuracy = (float)(((float)CCI/(float)testExamples)*100);
    
    cerr<<"-------Attribute Weighted-KNN---------------------------"<<endl;
    cerr<<"Number of Training Examples      # "<<trainExamples<<endl;
    cerr<<"Number of Testing Examples       # "<<testExamples<<endl;
    cerr<<"K used                           = "<<K<<endl;
    cerr<<"Correctly Classified Instances   # "<<CCI<<endl;
    cerr<<"Incorrectly Classified Instances # "<<testExamples - CCI<<endl;
    cerr<<"Accuracy (%)                     = "<<accuracy<<endl;
    cerr<<"--------------------------------------------------------"<<endl;
    
    return accuracy;
}

/*------------------------------------------------------------------*/
/* Test Simple KNN, Instance WeightedKNN, Attribute WeightedKNN and */
/* KNN by Backward Elimination)                                     */
/* tlist - training list                                            */
/* data - testing list                                              */
/* isInstanceWeighted - Instance WKNN                               */
/* mode - Training/Testing                                          */
/* isBackwardElimination - KNN by Backward Elimination              */
/* isAttWKNN - Attribute WKNN                                       */
/*------------------------------------------------------------------*/
int TestKNN (TRAINING_EXAMPLES_LIST *tlist, TRAINING_EXAMPLES_LIST data,
             bool isInstanceWeighted, MODE mode,
             bool isBackwardElimination, bool isAttWKNN)
{
    int correctlyClassifiedInstances = 0;
    TRAINING_EXAMPLES_LIST::iterator testIter;
    TrainingExample tmpTestObj;
    uint index[K];
    
    for (testIter = data.begin(); testIter != data.end(); ++testIter) {
        tmpTestObj = *testIter;
        /* Predict the class for the query point */
        int predictedClass = PredictByKNN(tlist, tmpTestObj.Value,
                                          isInstanceWeighted,
                                          index, mode, isBackwardElimination,
                                          isAttWKNN);
        
//        cerr << "Entry id: " << tmpTestObj.index << ", predicted class: " << predictedClass <<endl;
        
        /* Count the number of correctly classified instances */
        if ((int)tmpTestObj.Value[NO_OF_ATT - 1] == predictedClass) {
            correctlyClassifiedInstances ++;
        }
    }
    return correctlyClassifiedInstances;
}

/*------------------------------------------------------------------*/
/* Initialize attribute weights to random values                    */
/*------------------------------------------------------------------*/
void InitAttWeights ()
{
    srand (time(NULL));
    for(int i = 0; i < NO_OF_ATT - 1; i++)
        attWeights[i] = ((double)(rand () % 100 + 1))/100;
}


/*------------------------------------------------------------------*/
/* Normalize values by using mean and standard deviation            */
/*------------------------------------------------------------------*/
void NormalizeByStandardDeviation (TRAINING_EXAMPLES_LIST *trainList, int trainExamples) {
    
    for(int i = 0; i < NO_OF_ATT - 1; i++) {
        /* Find the mean */
        double mean = 0.0;
        TRAINING_EXAMPLES_LIST::iterator Iter;
        TrainingExample tmpTestObj;
        for(Iter = trainList->begin(); Iter != trainList->end(); ++Iter) {
            tmpTestObj = *Iter;
            mean += tmpTestObj.Value[i];
        }
        mean = (mean / (double)trainExamples);
        
        /* Find the deviation */
        double deviation = 0.0;
        for(Iter = trainList->begin(); Iter != trainList->end(); ++Iter) {
            tmpTestObj = *Iter;
            double val = mean - tmpTestObj.Value[i];
            deviation += val * val;
        }
        deviation = (deviation / (double)trainExamples);
        deviation = sqrt (deviation);
        
        /* Normalize the values */
        for(Iter = trainList->begin(); Iter != trainList->end(); ++Iter) {
            tmpTestObj = *Iter;
            double val = (tmpTestObj.Value[i] - mean)/deviation;
            Iter->Value[i] = val;
        }
    }
    
}

/*------------------------------------------------*/
/* Initialize the array to include all attributes */
/*------------------------------------------------*/
void BackwardEliminationInit ()
{
    for (int i = 0; i < NO_OF_ATT -1; i++)
        isBEAttIncluded[i] = true;
}

/*---------------------------------------------------------------------------*/
/* Predict Class by KNN (Simple KNN, Instance WeightedKNN,                   */
/*                       Attribute WeightedKNN, KNN by Backward Elimination) */
/* tlist - training list                                                     */
/* query - query instance to be classified                                   */
/* isWeightedKNN - Instance Weighted KNN                                     */
/* index - Indices of the K nearest neighbors will be stored in index array  */
/* mode - Specifies whether we are training or testing                       */
/* isBE - Backward Elimination                                               */
/* isAttWeightedKNN - Attribute WeightedKNN                                  */
/*---------------------------------------------------------------------------*/
int PredictByKNN (TRAINING_EXAMPLES_LIST *tlist, double *query,
                  bool isWeightedKNN, uint *index, MODE mode,
                  bool isBE, bool isAttWeightedKNN)
{
    
    //    cerr<<endl<<"PredictByKNN start"<<endl;
    
    double distance = 0.0;
    TRAINING_EXAMPLES_LIST::iterator iter;
    TrainingExample tmpObj;
    TRAINING_EXAMPLES_LIST elistWithD;
    
    if(!elistWithD.empty())
        elistWithD.clear ();
    
    /* If we are in for backward elimination or attribute WKNN */
    /* then Instance WKNN has to be false                      */
    if(isBE || isAttWeightedKNN)
        isWeightedKNN = false;
    
    /* Calculate the distance of the query */
    /* point from all training instances   */
    /* using the euclidean distance        */
    for(iter = tlist->begin(); iter != tlist->end(); ++iter)
    {
        tmpObj = *iter;
        distance = 0.0;
        
        for(int j = 0; j < NO_OF_ATT - 1; j++)
        {
            if(isAttWeightedKNN)
            {
                distance += (abs(query[j] - tmpObj.Value[j]) * abs(query[j] - tmpObj.Value[j])) * (attWeights[j] * attWeights[j]);
            }
            else
            {
                if(isBE)
                {
                    if(isBEAttIncluded[j])
                        distance += abs(query[j] - tmpObj.Value[j]) * abs(query[j] - tmpObj.Value[j]);
                }
                else
                {
                    if(isWeightedKNN)
                    {
                        if(isBEAttIncluded[j])
                            distance += abs(query[j] - tmpObj.Value[j]) * abs(query[j] - tmpObj.Value[j]);
                    }
                    else
                        distance += abs(query[j] - tmpObj.Value[j]) * abs(query[j] - tmpObj.Value[j]);
                }
            }
        }
        distance = sqrt(distance);
        /* If the distance is zero then set it to some high value */
        /* since it the query point itself                        */
        if((int)(distance*1000) == 0)
            distance = 999999999999999.9;
        
        tmpObj.Distance = distance;
        elistWithD.insert (elistWithD.end(), tmpObj);
    }
    
    //    cerr << "Query distance: " << distance << ", elistWithD size: " << elistWithD.size() << endl;
    
    /* Sort the points on distance in ascending order */
    elistWithD.sort(compare);
    
    if(!isWeightedKNN)
    {
        /* Simple KNN, Attribute Weighted KNN, Backward Elimination */
        int classCount[NO_OF_CLASSES];
        
        for(int i = 0; i < NO_OF_CLASSES; i++)
            classCount[i] = 0;
        
        int knn = K;
        for(iter = elistWithD.begin(); iter != elistWithD.end(); ++iter)
        {
            /* Calculate how the K nearest neighbors are classified */
            tmpObj = *iter;
            
            //            printExample(tmpObj);
            
            classCount[(int)tmpObj.Value[NO_OF_ATT-1]]++;
            knn--;
            if(!knn)
                break;
        }
        
        int maxClass = 0;
        int maxCount = 0;
        
        /* Find the class represented maximum number of times */
        /* among the k neighbors                              */
        for(int i = 0; i < NO_OF_CLASSES; i++)
        {
            if(classCount[i] > maxCount)
            {
                maxClass = i;
                maxCount = classCount[i];
            }
        }
        
        return maxClass;
    }
    else
    {
        /*Instance Weighted KNN */
        int knn = K;
        double pclass = 0.0;
        int i = 0;
        int maxClass = 0;
        /* Calulate the class by multiplying the K nearest  */
        /* neighbor weights by the class values.            */
        for(iter = elistWithD.begin(); iter != elistWithD.end(); ++iter)
        {
            tmpObj = *iter;
            
            /* While testing, do not use the training examples   */
            /* which were not near any instance during training. */
            if(mode == TESTING && tmpObj.isNearest2AtleastSome == false)
                continue;
            
            pclass += tmpObj.Weight * tmpObj.Value[NO_OF_ATT-1];
            /* Store the indices of the K nearest neighbors */
            index[i++] = tmpObj.index;
            knn--;
            if(!knn)
                break;
        }
        
        /* Mark an instance near when it is near to some query instance */
        for(iter = tlist->begin(); iter != tlist->end(); ++iter)
        {
            tmpObj = *iter;
            for(int i = 0; i < K; i++)
            {
                if(index[i] == tmpObj.index)
                {
                    iter->isNearest2AtleastSome = true;
                    break;
                }
            }
        }
        
        maxClass = (int)pclass;
        return maxClass;
    }
}

/*---------------------------------------------------------------------------*/
/* 3 Fold Cross Validation                                                   */
/* data - training data                                                      */
/* iterations - learn weights for number of iterations on a given cross fold */
/* numExamples - # of examples in the training set                           */
/* isAttWKNN = true (Learn attribute weights)                                */
/*           = false (Learn instance weights)                                */
/*---------------------------------------------------------------------------*/
float LearnWeights (TRAINING_EXAMPLES_LIST *tlist, //Training Data
                    TRAINING_EXAMPLES_LIST data,   //Testing Data
                    //Train data = Test data
                    int iterations,				   //Learn for # of iterations
                    int numExamples,			   //# of examples
                    MODE mode,					   // mode = TRAINING
                    int desiredAccuracy,		   //Desired accuracy in %
                    bool isAttWeightedKNN)	   //Attribute or Instance Weighted
{
    TRAINING_EXAMPLES_LIST::iterator iter;
    uint index[K];
    int CCI;
    float accuracy = 0;
    
    /* Learn weights for number of iterations  */
    /* or till the desired accuracy is reached */
    int tmp = 0;
    while(tmp!=iterations)
    {
        for(iter = data.begin(); iter != data.end(); ++iter)
        {
            TrainingExample TEObj = *iter;
            /* Predict the class */
            int predictedClass = PredictByKNN (tlist, TEObj.Value, true, index,
                                               mode, false, isAttWeightedKNN);
            int actualClass = (int)(TEObj.Value[NO_OF_ATT-1]);
            if(actualClass != predictedClass)
            {
                /* Calculate the error */
                double error = actualClass - predictedClass;
                /* Adjust weights by Gradient Descent */
                AdjustWeightsByGradientDescent (TEObj.Value, tlist, error,
                                                index,isAttWeightedKNN);
            }
        }
        
        CCI = TestKNN (tlist, data, true, mode, false, false);
        accuracy = (float)(((float)CCI/(float)numExamples)*100);
        tmp++;
        
        int iacc = (int)accuracy;
        if(iacc > desiredAccuracy)
            break;
        cerr << "LearnWeights accuracy: " << accuracy << " at iteration: " << tmp << endl;
    }
    return accuracy;
}

/*---------------------------------------------------------------------------*/
/* 3 Fold Cross Validation                                                   */
/* data - training data                                                      */
/* iterations - learn weights for number of iterations on a given cross fold */
/* numExamples - # of examples in the training set                           */
/* isAttWKNN = true (Learn attribute weights)                                */
/*           = false (Learn instance weights)                                */
/*---------------------------------------------------------------------------*/
float CrossValidate(TRAINING_EXAMPLES_LIST *data, int iterations,
                    int numExamples, bool isAttWKNN)
{
    /* Divide the data into three equal sets N1,N2,N3   */
    /* First Cross Fold:  Training = N1+N2 Testing = N3 */
    /* Second Cross Fold: Training = N2+N3 Testing = N1 */
    /* Third Cross Fold:  Training = N1+N3 Testing = N2 */
    int N = numExamples/3;
    int first = N;
    int second = 2*N;
    int i = 0;
    double accuracy = 0.0;
    
    TRAINING_EXAMPLES_LIST trainList,testList;
    TRAINING_EXAMPLES_LIST::iterator iter;
    
    /* first cross fold validation */
    i = 0;
    for(iter = data->begin(); iter != data->end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        if(i < second)
            trainList.insert (trainList.end (), TEObj);
        else
            testList.insert (testList.end (), TEObj);
        i++;
    }
    
    /* Learn Weights (Test and adjust by gradient descent) */
    accuracy = LearnWeights (&trainList, testList, iterations,
                             second + 1, TRAINING, 95, isAttWKNN);
    
    data->clear ();
    for(iter = trainList.begin(); iter != trainList.end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        data->insert (data->end (), TEObj);
    }
    
    for(iter = testList.begin(); iter != testList.end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        data->insert (data->end (), TEObj);
    }
    
    /* second cross fold validation */
    trainList.clear ();
    testList.clear();
    i = 0;
    for(iter = data->begin(); iter != data->end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        if(i >= first)
            trainList.insert (trainList.end (), TEObj);
        else
            testList.insert (testList.end (), TEObj);
        
        i++;
    }
    
    /* Learn Weights (Test and adjust by gradient descent) */
    accuracy = LearnWeights (&trainList, testList, iterations,
                             numExamples-first + 1, TRAINING, 95, isAttWKNN);
    data->clear ();
    for(iter = testList.begin(); iter != testList.end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        data->insert (data->end (), TEObj);
    }
    
    for(iter = trainList.begin(); iter != trainList.end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        data->insert (data->end (), TEObj);
    }
    
    
    /* third fold cross validation */
    trainList.clear ();
    testList.clear();
    
    i = 0;
    for(iter = data->begin(); iter != data->end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        if(i < first)
            trainList.insert (trainList.end (), TEObj);
        else if (i >= first && i < second)
            testList.insert (testList.end (), TEObj);
        else if (i >= second && i < numExamples)
            trainList.insert (trainList.end (), TEObj);
        i++;
    }
    
    /* Learn Weights (Test and adjust by gradient descent) */
    accuracy = LearnWeights (&trainList, testList, iterations,
                             first+numExamples-second + 1, TRAINING, 95, isAttWKNN);
    
    data->clear ();
    for(iter = trainList.begin(); iter != trainList.end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        data->insert (data->end (), TEObj);
    }
    
    for(iter = testList.begin(); iter != testList.end(); ++iter)
    {
        TrainingExample TEObj = *iter;
        data->insert (data->end (), TEObj);
    }
    
    return (float)accuracy;
}

/*-------------------------------------------------------------------------*/
/* qvalue - Value of a particular attribute                                */
/* tlist  - List of training examples                                      */
/* error  - Error in prediction of class for an instance                   */
/* index  - indices corresponding to the K nearest neighbors(instances)    */
/* isAttWeightedKNN = true (Adjust attribute weights)                      */
/*                  = false (Adjust instance weights)                      */
/*-------------------------------------------------------------------------*/
void AdjustWeightsByGradientDescent (double *qvalue,
                                     TRAINING_EXAMPLES_LIST *tlist,
                                     double error, uint *index,
                                     bool isAttWeightedKNN)
{
    if(isAttWeightedKNN)
    {
        /* Adjust attribute weights by gradient descent*/
        for(int i = 0; i < NO_OF_ATT - 1; i++)
            attWeights[i] = attWeights[i] + LEARNING_RATE * error * qvalue[i];
    }
    else
    {
        /* Adjust instance weights by gradient descent.     */
        /* We adjust the weights of the K nearest neighbors */
        /* for a query instance                             */
        TRAINING_EXAMPLES_LIST::iterator iter;
        int k = K;
        
        for(iter = tlist->begin(); iter != tlist->end(); ++iter)
        {
            TrainingExample TEObj = *iter;
            for(int i = 0; i < K; i++)
            {
                if(TEObj.index == index[i])
                {
                    double wt = TEObj.Weight;
                    wt = wt + LEARNING_RATE * error;
                    iter->Weight = wt;
                    k--;
                    break;
                }
            }
            if(k == 0)
                break;
        }
    }
}

/*-------------------------------------------------------------------------*/
/* Comparison function used by the sorting function for list objects.      */
/*-------------------------------------------------------------------------*/
bool compare(const TrainingExample t1, const TrainingExample t2)
{
    if (t1.Distance < t2.Distance)
        return true;
    else
        return false;
}

/*------------------------------------------------------------------------*/
/* line - to read file fp line                                            */
/* max - maximum line length to read                                      */
/* fp - file to read from                                                 */
/* Return Parameter: 0 if end of file, line length otherwise.             */
/* Copies a file contents to another file.                                */
/*------------------------------------------------------------------------*/
int GetLine (char *line, int max, FILE *fp) {
    if(fgets(line, max, fp)==NULL)
        return 0;
    else
        return strlen(line);
}

/*-----------------------------------------------------------------*/
/* filename - File from which the training/testing data is read    */
/* rlist - The data structure that holds the training/test data    */
/* rlistExamples - # of training/test examples                     */
/*-----------------------------------------------------------------*/
bool readData4File (char *filename, TRAINING_EXAMPLES_LIST *rlist,
                    int *rlistExamples)
{
    FILE *fp = NULL;
    int len = 0;
    char line[LINE_MAX+1];
    int lineSize = LINE_MAX;
    TrainingExample *TEObj;
    int index = 0;
    int numExamples = 0;
    
    *rlistExamples = 0;
    
    line[0] = 0;
    
    if((fp = fopen (filename, "r")) == NULL)
    {
        cout<<"Error in opening file."<<endl;
        return false;
    }
    
    //Initialize weights to random values
    srand (time(NULL));
    
    char *tmp;
    int tmpParams = 0; //NO_OF_ATT;
    double cd = 0.0;
    
    /* Read the data file line by line */
    while((len = GetLine (line, lineSize, fp))!=0)
    {
        TEObj = new TrainingExample ();
        tmp = strtok (line,",");
        while (tmp != NULL)
        {
            cd = atof (tmp);
            TEObj->Value[tmpParams] = cd;
            tmpParams ++;
            
            tmp = strtok (NULL, ",");
            
            if(tmpParams == NO_OF_ATT)
            {
                tmpParams = 0;
                cd = 0.0;
                line[0] = 0;
                numExamples ++;
                
                //Not using this normalization anymore.
                // N(y) = y/(1+y)
                // Doing normalization by standard deviation and mean
                //TEObj->NormalizeVals ();
                
                /* Generating random weights for instances. */
                /* These weights are used in instance WKNN  */
                double rno = (double)(rand () % 100 + 1);
                TEObj->Weight = rno/100;
                TEObj->index = index++;
                TEObj->isNearest2AtleastSome = false;
                break;
            }
        }
        
        rlist->insert (rlist->end(), *TEObj);
        
        delete TEObj;
    }
    
    /* Normalize values using standard deviation */
    NormalizeByStandardDeviation (rlist, numExamples);
    
    *rlistExamples = numExamples;
    
    return true;
}
