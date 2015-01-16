//
//  ActiveMolecules.h
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/16/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//

#ifndef activemoleculesC___ActiveMolecules_h
#define activemoleculesC___ActiveMolecules_h

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
                    features.PB(f);
                    if (f == NVL) {
                        completeFeatures = false;
                        // store missed feature
                        missedFeatures.PB(index);
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
        
        int indexes[] = {2, 7, 9, 12, 5, 16, 8, 6, 18, 20, 14, 1, 15, 11, 17, 0, 10, 19, 4}; // 901736.64
        
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

class ActiveMolecules {
    int X;
    int Y;
    int M;
    std::vector<std::vector<double>>matrix;
    
public:
    virtual int similarity(int moleculeID, std::vector<double> &similarities);
    std::vector<int> rank(const std::vector<std::string> &train, const std::vector<std::string> &test);
    
private:
#ifdef USE_XGBOOST
    std::vector<int> renkByXGBRegression(std::vectorEntry> &training, std::vectorEntry> &testing);
};

#endif
