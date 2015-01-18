//
//  TestRunner.cpp
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/16/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//
#include <iostream>

#define FROM_TEST_RUNNER

#include "stdc++.h"
#include "ActiveMolecules.cpp"

namespace am {
    namespace launcher {
        
        /*
         * The random class generator.
         */
        struct SecureRandom {
            int seed;
            
            SecureRandom(int seed) : seed(seed){}
            
            int nextInt(int max) {
                std::default_random_engine engine(seed);
                std::uniform_int_distribution<int> distribution(0, max - 1);
                
                return distribution(engine);
            }
        };
        
        SecureRandom rnd(1);
        
        VS splt(std::string s, char c = ',') {
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
        
        class ActiveMoleculesVis {

            const int NUM_TEST_MOLECULES = 200;
            
            int numTrainingData;
            int numExtraTrainingData;
            
            int X, Y;
            VVD S;
            VD P;
            VS DTrain;
            VS DTest;
            VI gtfRank;
            
            ActiveMolecules *test;
            
        public:
            std::string similarityFile = "./tester/data/example_s.csv";
            std::string dataFile = "./tester/data/example_data.csv";
            
            ActiveMoleculesVis(ActiveMolecules *test) : test(test) {}
            
            double doExec() {
                numExtraTrainingData = 0;
                numTrainingData = 3000;
                
                if (!generateTest()) {
                    std::cerr << "Failed to generate test data" << std::endl;
                    return -1;
                }
                
                //
                // Start solution testing
                //
                
                for (int i = 0; i < X + Y; i++) {
                    VD similarities;
                    for (int j = 0; j < X + Y; j++) {
                        similarities.push_back(S[j][i]);
//                        similarities.push_back(S[i][j]);
                    }
                    test->similarity(i, similarities);
                }
                
                // call rank
                
                VI userAns = test->rank(DTrain, DTest);
                VI used(Y, 0);
                
                for (int i = 0; i < Y; i++) {
                    userAns[i] = userAns[i] - X;
                    if (userAns[i] < 0 || userAns[i] >= Y) {
                        fprintf(stderr, "ERROR: Value in return out of range: %i", userAns[i] + X);
                        return 0.0;
                    }
                    if (used[userAns[i]] != 0) {
                        fprintf(stderr, "ERROR: Duplicate value in return: %i", (userAns[i] + X));
                        return 0.0;
                    }
                    used[userAns[i]] = 1;
                }
                
                double score = 0.0;
                double count = 0.0;
                
                for (int i = 0; i < Y; i++) {
                    count = 0;
                    for (int j = 0; j <= i; j++) {
                        if (gtfRank[userAns[j]] <= i) count++;
                    }
                    score += count / (i + 1);
                }
                
                score *= 1000000.0 / Y;
                
                return score;
                
            }
            
        private:
            
            /**
             * generates test data.
             *
             * return false in case of error.
             */
            bool generateTest() {
                X = numTrainingData + numExtraTrainingData;
                Y = NUM_TEST_MOLECULES;
                
                fprintf(stderr, "Similarity file: %s, data file: %s, train size: %i, test size: %i\n", similarityFile.c_str(), dataFile.c_str(), X, Y);
                
                // read similarity
                std::ifstream simfile (similarityFile);
                if (!simfile.is_open()) {
                    std::cerr << "Error in opening file: " << similarityFile << std::endl;
                    return false;
                }
                
                std::string line;
                int T = 0;
                if (!getline(simfile, line) || sscanf(line.c_str(), "%i", &T) != 1) {
                    std::cerr << "Failed to read T" << std::endl;
                } else {
                    fprintf(stderr, "Expected records count: %i\n", T);
                }
                
                VI trainID(X, 0);
                VI testID(Y, 0);
                VI selected(T, 0);
                
                // base training data
                for (int i = 0; i < numTrainingData; i++) {
                    selected[i] = 1;
                    trainID[i] = i;
                }
                // select test data
                RandomSample sample(T - X, Y);
                VI indexes = sample.get_sample_index();
                for (int i = 0; i < Y; i++) {
                    int index = indexes[i] + X;
//                    fprintf(stderr, "index: %i, i: %i\n", index, i);
                    if (selected[index]) {
                        cerr << "Duplicate selected index: " << index << endl;
                    }
                    selected[index] = 1;
                }
                
//                S = new double[X + Y][X + Y];

                int r_idx = 0;
                for (int row = 0; row < T; row++) {
                    getline(simfile, line);
                    if (selected[row] != 1) {
                        continue; // skip these rows, they are not used for the test
                    }
                    
//                    fprintf(stderr, "%s\n", line.c_str());
//                    fprintf(stderr, "Row #%i\n", row);
                    
                    VS items = splt(line, ',');
                    VD rowVals;
                    for (int i = 0; i < T; i++) {
                        if (selected[i] == 1) {
                            rowVals.push_back(std::stod(items[i]));
                        }
                    }
                    S.push_back(rowVals);
                    r_idx++;
                }
                simfile.close();
                
                cerr << "Similarity data collected" << endl;
                
                //
                // load data
                //
                std::ifstream datafile (dataFile);
                if (!datafile.is_open()) {
                    std::cerr << "Error in opening file: " << dataFile << std::endl;
                    return false;
                }
                r_idx = 0;
                for (int row = 0; row < T; row++) {
                    getline(datafile, line);
                    if (selected[row] != 1) continue; // skip these rows, they are not used for the test
                    
                    VS items = splt(line, ',');
                    P.push_back(std::stod(items[items.size() - 1]));
                    if (r_idx < X) {
                        DTrain.push_back(line);
                    } else {
                        std::string snew = items[0];
                        for (int i = 1; i < items.size() - 1; i++)
                            snew += "," + items[i];
                        
                        DTest.push_back(snew);
                    }
                    r_idx++;
                }
                datafile.close();
                
                cerr << "Main data collected" << endl;
                
                // create ground truth ranked list
                gtfRank.resize(Y, 0);
                for (int i = 0; i < Y; i++) {
                    gtfRank[i] = 0;
                    for (int j = 0; j < Y; j++)
                        if (i != j) {
                            if (P[X + j] > P[X + i]) gtfRank[i]++;
                        }
                }
                
                cerr << "Ground truth ranked list created" << endl;
                
                // everything OK
                return true;
            }
        };
    }
}

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        printf("Usage: similaritiesFile dataFile\n");
        return 0;
    }
    ActiveMolecules task;
    am::launcher::ActiveMoleculesVis runner(&task);
    runner.similarityFile = argv[1];
    runner.dataFile = argv[2];
    
    int tests = 1;
    double mean = 0;
    
    for (int i = 0; i < tests; i++) {
        double score = runner.doExec();
        fprintf(stderr, "%i.) Score  = %f\n", i, score);
    }
    mean /= tests;
    fprintf(stderr, "Mean score: %f for %i tests\n", mean, tests);
    return 0;
}

