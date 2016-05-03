## Overwiev ##
Provided with set of data describing complex molecules and experimentaly detected reactions with antigens the main task 
was to build and train data model able to rank new data sets in accordance with similarity to known molecules.
Thus it would be possible to make angtigen-molecule simulations instead of costly experiments.
It was done as part of crowdsourcing contest on TopCoder: [Contest: Active Molecules](https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16275)

### Special conditions ###
Additional requirement was to create crafted algorithm which may be compilled and runned in the AWS runner. With specified 
limitation in time of compilation, running and memory consumption.

The main algorithm runner implemented in: [ActiveMolecules.cpp](https://github.com/yaricom/activemolecules-c/blob/master/activemoleculesC%2B%2B/ActiveMolecules.cpp)
