//
//  TestRunner.cpp
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/16/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//
#include <iostream>

#include "stdc++.h"
#include "ActiveMolecules.h"

#define PB          push_back
#define LL          long long
#define ULL         unsigned long long
#define LD          long double
#define MP          std::make_pair
#define VC          std::vector
#define PII         std::pair <int, int>
#define VI          VC < int >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VPII        VC < PII >
#define VD          VC < double >
#define VVD         VC < VD >
#define VF          VC < float >
#define VVF         VC < VF >
#define VS          VC < std::string >
#define VVS         VC < VS >

namespace am {
namespace launcher {
    
    const static std::string SIMILARITY_FILE = "./tester/data/example_s.csv";
    const static std::string DATA_FILE = "./tester/data/example_data.csv";
    
    const static int NUM_TEST_MOLECULES = 200;
    
    static int numTrainingData;
    static int numExtraTrainingData;
    
    static int X, Y;
    static VVD S;
    static VD P;
    static VS DTrain;
    static VS DTest;
    static VI gtfRank;
    
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
                all.PB(s.substr(p, np - p));
            else
                all.PB("");
            
            p = np + 1;
        }
        if (p < s.size())
            all.PB(s.substr(p));
        return all;
    }
    
    /**
     * generates test data.
     *
     * return false in case of error.
     */
    bool generateTest() {
        X = numTrainingData + numExtraTrainingData;
        Y = NUM_TEST_MOLECULES;
        
        // read similarity
        std::ifstream simfile (SIMILARITY_FILE);
        if (!simfile.is_open()) {
            std::cerr << "Error in opening file: " << SIMILARITY_FILE << std::endl;
            return false;
        }
        
        std::string line;
        int T = 0;
        if (!getline(simfile, line) || sscanf(line.c_str(), "%i", &T) != 1) {
            std::cerr << "Failed to read T" << std::endl;
        } else {
            fprintf(stderr, "Expected records count: $i", T);
        }
        
        VI trainID(X, 0);
        VI testID(Y, 0);
        VI selected(T, 0);
        
        // base training data
        for (int i = 0; i < numTrainingData; i++) {
            selected[i] = 1;
            trainID[i] = i;
        }
        // select extra training data
        for (int i = numTrainingData; i < X; i++) {
            
            int y;
            do {
                y = rnd.nextInt(T);
            } while (selected[y] != 0);
            selected[y] = 1;
            trainID[i] = y;
        }
        // select test data
        for (int i = 0; i < Y; i++) {
            int y;
            do {
                y = rnd.nextInt(T);
            } while (selected[y] != 0);
            selected[y] = 1;
            testID[i] = y;
        }
        int r_idx = 0;
        for (int row = 0; row < T; row++) {
            getline(simfile, line);
            if (selected[row] != 1) {
                continue; // skip these rows, they are not used for the test
            }
            VS items = splt(line, ',');
            int c_idx = 0;
            for (int i = 0; i < T; i++)
                if (selected[i] == 1) {
                    S[c_idx++][r_idx] =  std::stod(items[i]);
                }
            r_idx++;
        }
        simfile.close();
        
        //
        // load data
        //
        std::ifstream datafile (DATA_FILE);
        if (!simfile.is_open()) {
            std::cerr << "Error in opening file: " << DATA_FILE << std::endl;
            return false;
        }
        r_idx = 0;
        for (int row = 0; row < T; row++) {
            getline(datafile, line);
            if (selected[row] != 1) continue; // skip these rows, they are not used for the test
            
            VS items = splt(line, ',');
            P[r_idx] = std::stod(items[items.size() - 1]);
            if (r_idx < X) {
                DTrain[r_idx] = line;
            } else {
                std::string snew = items[0];
                for (int i = 1; i < items.size() - 1; i++)
                    snew += "," + items[i];
                
                DTest[r_idx - X] = snew;
            }
            r_idx++;
        }
        datafile.close();
     
        // create ground truth ranked list
        for (int i = 0; i < Y; i++) {
            gtfRank[i] = 0;
            for (int j = 0; j < Y; j++)
                if (i != j) {
                    if (P[X + j] > P[X + i]) gtfRank[i]++;
                }
        }
        
        // everything OK
        return true;
    }
    
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
        ActiveMolecules test;

        for (int i = 0; i < X + Y; i++) {
            VD similarities;
            for (int j = 0; j < X + Y; j++) {
                similarities[j] = S[j][i];
            }
            test.similarity(i, similarities);
        }
        
        // call rank
        
        VI userAns = test.rank(DTrain, DTest);
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
}
}

int main(int argc, const char * argv[]) {
    int tests = 1;
    double mean = 0;
    for (int i = 0; i < tests; i++) {
        double score = am::launcher::doExec();
        fprintf(stderr, "%i.) Score  = %f\n", i, score);
    }
    mean /= tests;
    fprintf(stderr, "Mean score: %f for %i tests", mean, tests);
    return 0;
}

