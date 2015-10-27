cmake_minimum_required(VERSION 2.8.3)
project(appl_offline)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")

include_directories(
  src/Algorithms
  src/Algorithms/HSVI
  src/Algorithms/SARSOP
  src/Bounds
  src/Controller
  src/Core
  src/Evaluator
  src/MathLib
  src/Models/MOMDP
  src/Models/MOMDP/CoLoc
  src/OfflineSolver
  src/Parser
  src/Parser/Cassandra
  src/Parser/Cassandra/include
  src/Parser/POMDPX
  src/Simulator
  src/Utils
)

add_library("${PROJECT_NAME}" SHARED
  src/Algorithms/SARSOP/BinManager.cpp
  src/Algorithms/SARSOP/BinManagerSet.cpp
  src/Algorithms/SARSOP/SARSOP.cpp
  src/Algorithms/SARSOP/SARSOPPrune.cpp
  src/Algorithms/SARSOP/Sample.cpp
  src/Algorithms/SARSOP/SampleBP.cpp
  src/Bounds/AlphaPlane.cpp
  src/Bounds/AlphaPlanePool.cpp
  src/Bounds/AlphaPlanePoolSet.cpp
  src/Bounds/AlphaVectorPolicy.cpp
  src/Bounds/BackupAlphaPlaneMOMDP.cpp
  src/Bounds/BackupAlphaPlaneMOMDPLite.cpp
  src/Bounds/BackupBeliefValuePairMOMDP.cpp
  src/Bounds/BackupBeliefValuePairMOMDPLite.cpp
  src/Bounds/BeliefValuePair.cpp
  src/Bounds/BeliefValuePairPool.cpp
  src/Bounds/BeliefValuePairPoolSet.cpp
  src/Bounds/BlindLBInitializer.cpp
  src/Bounds/FastInfUBInitializer.cpp
  src/Bounds/FullObsUBInitializer.cpp
  src/Bounds/PruneAlphaPlane.cpp
  src/Bounds/PruneBeliefValuePair.cpp
  src/Bounds/xml_parse_lib.c
  src/Controller/Controller.cpp
  src/Core/Actions.cpp
  src/Core/Belief.cpp
  src/Core/BeliefCache.cpp
  src/Core/BeliefException.cpp
  src/Core/BeliefForest.cpp
  src/Core/BeliefTreeNode.cpp
  src/Core/BeliefWithState.cpp
  src/Core/MObject.cpp
  src/Core/Observations.cpp
  src/Core/States.cpp
  src/Core/UniqueBeliefHeap.cpp
  src/Core/VariableContainer.cpp
  src/Evaluator/EvaluationEngine.cpp
  src/Evaluator/EvaluatorSampleEngine.cpp
  src/MathLib/DenseVector.cpp
  src/MathLib/MathLib.cpp
  src/MathLib/SparseMatrix.cpp
  src/MathLib/SparseVector.cpp
  src/Models/MOMDP/BeliefTransitionMOMDP.cpp
  src/Models/MOMDP/BeliefTransitionMOMDPLite.cpp
  src/Models/MOMDP/BooleanVariable.cpp
  src/Models/MOMDP/IVariable.cpp
  src/Models/MOMDP/IVariableValue.cpp
  src/Models/MOMDP/MOMDP.cpp
  src/Models/MOMDP/MOMDPLite.cpp
  src/Models/MOMDP/ObservationProbabilities.cpp
  src/Models/MOMDP/Rewards.cpp
  src/Models/MOMDP/StateTransitionX.cpp
  src/Models/MOMDP/StateTransitionXXpY.cpp
  src/Models/MOMDP/StateTransitionXY.cpp
  src/Models/MOMDP/StateTransitionY.cpp
  src/Models/MOMDP/Variable.cpp
  src/Models/MOMDP/VariableCombined.cpp
  src/Models/MOMDP/VariableRelation.cpp
  src/Models/MOMDP/VariableValue.cpp
  src/OfflineSolver/GlobalResource.cpp
  src/OfflineSolver/solverUtils.cpp
  src/Parser/Cassandra/POMDP.cpp
  src/Parser/Cassandra/Parser.cpp
  src/Parser/Cassandra/decision-tree.c
  src/Parser/Cassandra/imm-reward.c
  src/Parser/Cassandra/include/pomdp_spec.tab.cc
  src/Parser/Cassandra/include/pomdp_spec.yy.cc
  src/Parser/Cassandra/mdpCassandra.c
  src/Parser/Cassandra/parse_err.c
  src/Parser/Cassandra/parse_hash.c
  src/Parser/Cassandra/pomdpCassandraWrapper.cpp
  src/Parser/Cassandra/sparse-matrix.c
  src/Parser/POMDPX/FactoredPomdp.cpp
  src/Parser/POMDPX/Function.cpp
  src/Parser/POMDPX/ObsAct.cpp
  src/Parser/POMDPX/PreCEntry.cpp
  src/Parser/POMDPX/PreCMatrix.cpp
  src/Parser/POMDPX/SparseEntry.cpp
  src/Parser/POMDPX/SparseTable.cpp
  src/Parser/POMDPX/State.cpp
  src/Parser/POMDPX/StateObsAct.cpp
  src/Parser/POMDPX/tinystr.cpp
  src/Parser/POMDPX/tinyxml.cpp
  src/Parser/POMDPX/tinyxmlerror.cpp
  src/Parser/POMDPX/tinyxmlparser.cpp
  src/Parser/ParserSelector.cpp
  src/PolicyGraph/PolicyGraph.cpp
  src/PolicyGraph/PolicyGraphGenerator.cpp
  src/Simulator/SimulationEngine.cpp
  src/Utils/CPMemUtils.cpp
  src/Utils/InfoLog.cpp
  src/Utils/SimulationRewardCollector.cpp
  src/Utils/StatsCollector.cpp
  src/Utils/md5.cpp
  src/Utils/md5wrapper.cpp
  src/miniposix/getopt.c
  src/miniposix/getopt1.c
  src/miniposix/getopt_init.c
)

add_executable("${PROJECT_NAME}_pomdpeval" src/Evaluator/Evaluator.cpp)
target_link_libraries("${PROJECT_NAME}_pomdpeval" ${PROJECT_NAME})

add_executable("${PROJECT_NAME}_pomdpsim" src/Simulator/Simulator.cpp)
target_link_libraries("${PROJECT_NAME}_pomdpsim" ${PROJECT_NAME})

add_executable("${PROJECT_NAME}_pomdpsol" src/OfflineSolver/solver.cpp)
target_link_libraries("${PROJECT_NAME}_pomdpsol" ${PROJECT_NAME})

add_executable("${PROJECT_NAME}_polgraph" src/PolicyGraph/PolicyGraph.cpp)
target_link_libraries("${PROJECT_NAME}_polgraph" ${PROJECT_NAME})

add_executable("${PROJECT_NAME}_pomdpconvert" src/PomdpConvertor/convertor.cpp)
target_link_libraries("${PROJECT_NAME}_pomdpconvert" ${PROJECT_NAME})

install(TARGETS
    "${PROJECT_NAME}"
    "${PROJECT_NAME}_pomdpeval"
    "${PROJECT_NAME}_pomdpsim"
    "${PROJECT_NAME}_pomdpsol"
    "${PROJECT_NAME}_pomdpgraph"
    "${PROJECT_NAME}_pomdpconvert"
  EXPORT "${PROJECT_NAME}Targets"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY src
  DESTINATION include/appl/offline
  FILES_MATCHING PATTERN "*.h"
)
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)
