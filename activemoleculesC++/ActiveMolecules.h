//
//  ActiveMolecules.h
//  activemoleculesC++
//
//  Created by Iaroslav Omelianenko on 1/16/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//

#ifndef activemoleculesC___ActiveMolecules_h
#define activemoleculesC___ActiveMolecules_h



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
