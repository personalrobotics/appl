#include "belief_update/belief_update_exact.h"
#include "belief_update/belief_update_particle.h"
#include "globals.h"
#include "lower_bound/lower_bound_policy_mode.h"
#include "lower_bound/lower_bound_policy_random.h"
#include "model.h"
#include "optionparser.h"
#include "problems/bridge/bridge.h"
#include "problems/lasertag/lasertag.h"
#include "problems/pocman/belief_update_pocman.h"
#include "problems/pocman/pocman.h"
#include "problems/rocksample/rocksample.h"
#include "problems/tag/belief_update_tag.h"
#include "problems/tag/tag.h"
#include "problems/tiger/tiger.h"
#include "solver.h"
#include "upper_bound/upper_bound_nonstochastic.h"
#include "upper_bound/upper_bound_stochastic.h"
#include "world.h"

using namespace std;

enum optionIndex { 
  UNKNOWN, HELP, PROBLEM, SIZE, NUMBER, DEPTH, DISCOUNT, SEED, TIMEOUT, 
  NPARTICLES, REGPARAM, SIMLEN, LBTYPE, BELIEF, KNOWLEDGE, APPROX_BOUND
};

// option::Arg::Required is a misleading name. The program won't complain 
// if these are absent, so mandatory flags must be checked for presence
// manually.
const option::Descriptor usage[] = {
 {UNKNOWN,       0, "",  "", option::Arg::None, "USAGE: despot [options]\n\nOptions:"},
 {HELP,          0, "",  "help", option::Arg::None, "  --help   \tPrint usage and exit."},
 {PROBLEM,       0, "p", "problem", option::Arg::Required, "  -p <arg> \t--problem=<arg>  \tProblem name."},
 {SIZE,          0, "s", "size", option::Arg::Required, "  -s <arg> \t--size=<arg>  \tSize of problem (problem-specific)."},
 {NUMBER,        0, "n", "number", option::Arg::Required, "  -n <arg> \t--number=<arg>  \tNumber of elements in problem (problem-specific)."},
 {DEPTH,         0, "d", "depth", option::Arg::Required, "  -d <arg> \t--depth=<arg>  \tMaximum depth of search tree (default: 90)."},
 {DISCOUNT,      0, "g", "discount", option::Arg::Required, "  -g <arg> \t--discount=<arg>  \tDiscount factor (default: 0.95)."},
 {SEED,          0, "x", "seed", option::Arg::Required, "  -x <arg> \t--seed=<arg>  \tRandom number seed (default: 42)."},
 {TIMEOUT,       0, "t", "timeout", option::Arg::Required, "  -t <arg> \t--timeout=<arg>  \tSearch time per move, in seconds (default: 1)."},
 {NPARTICLES,    0, "k", "nparticles", option::Arg::Required, "  -k <arg> \t--nparticles=<arg>  \tNumber of particles (default: 500)."},
 {REGPARAM,      0, "r", "regparam", option::Arg::Required, "  -r <arg> \t--regparam=<arg>  \tRegularization parameter (default: no regularization)."},
 {SIMLEN,        0, "l", "simlen", option::Arg::Required, "  -l <arg> \t--simlen=<arg>  \tNumber of steps to simulate. (default: -1 = infinite)."},
 {LBTYPE,        0, "w", "lbtype", option::Arg::Required, "  -w <arg> \t--lbtype=<arg>  \tLower bound strategy, if applicable."},
 {BELIEF,        0, "b", "belief", option::Arg::Required, "  -b <arg> \t--belief=<arg>  \tBelief-update strategy (default: 'particle')."},
 {KNOWLEDGE,     0, "j", "knowledge", option::Arg::Required, "  -j <arg> \t--knowledge=<arg>  \tKnowledge level for random lower bound policy, if applicable."},
 {APPROX_BOUND, 0, "a", "approx-upper-bound", option::Arg::None, "  -a \t--approx-upper-bound  \tWhether the initial upper bounds is approximate or true (default: false)."},
 {0,0,0,0,0,0}
};

template<typename T>
bool AssignLowerBound(ILowerBound<T>*& lb_ptr, string problem, string lb_type,
    Model<T>* model, const RandomStreams& streams, option::Option* options) {

  static MAP<string, list<string>> valid_lower_bounds = {
    {"tag",      {"mode", "random"}},
    {"lasertag", {"mode", "random"}},
    {"tiger",    {"random"}},
    {"bridge",   {"random"}},
    {"pocman",   {"random"}},
  };

  auto& valid_types = valid_lower_bounds[problem];
  if (find(valid_types.begin(), valid_types.end(), lb_type) == valid_types.end()) {
    cout << "Invalid lower bound type " << lb_type << " for problem "
         << problem << endl;
    return false;
  }
  if (lb_type == "mode")
    lb_ptr = new ModePolicyLowerBound<T>(streams, model->NumStates());
  else if (lb_type == "random") {
    int knowledge = options[KNOWLEDGE] ? atoi(options[KNOWLEDGE].arg) : 2;
    lb_ptr = new RandomPolicyLowerBound<T>(streams, knowledge);
  }

  return true;
}

template<>
bool AssignLowerBound<RockSampleState>(ILowerBound<RockSampleState>*& lb_ptr,
    string problem, string lb_type, Model<RockSampleState>* model, 
    const RandomStreams& streams, option::Option* options) {
  lb_ptr = (RockSample*)model;
  return true;
}

template<typename T>
void AssignUpperBound(IUpperBound<T>*& ub_ptr, const RandomStreams& streams,
                      Model<T>* model) {
  ub_ptr = new UpperBoundStochastic<T>(streams, *model);
}

template<>
void AssignUpperBound<PocmanState>(IUpperBound<PocmanState>*& ub_ptr, 
    const RandomStreams& streams, Model<PocmanState>* model) {
  ub_ptr = (Pocman*)model;
}

template<>
void AssignUpperBound<RockSampleState>(IUpperBound<RockSampleState>*& ub_ptr, 
    const RandomStreams& streams, Model<RockSampleState>* model) {
  ub_ptr = new UpperBoundNonStochastic<RockSampleState>(streams, *model);

  // RockSample's custom lower bound needs this
  ((RockSample*)model)->set_upper_bound_act(
      static_cast<UpperBoundNonStochastic<RockSampleState>*>(ub_ptr)
        ->UpperBoundAct());
}

template<typename T>
bool AssignBeliefUpdate(BeliefUpdate<T>*& bu_ptr, string bu_type, 
    string problem, Model<T>* model, const RandomStreams& streams) {

  static MAP<string, list<string>> valid_belief_updates = {
    {"tag",        {"particle"}},
    {"lasertag",   {"particle"}},
    {"rocksample", {"particle", "exact"}},
    {"tiger",      {"particle"}},
    {"bridge",     {"particle"}},
    {"pocman",     {"particle"}},
  };

  if (bu_type == "particle")
    bu_ptr = new ParticleFilterUpdate<T>(streams.BeliefUpdateSeed(), *model);
  else if (bu_type == "exact")
    bu_ptr = new ExactBeliefUpdate<T>(streams.BeliefUpdateSeed(), *model);
  else {
    cerr << "belief update type " << bu_type << " not compatible with problem "
         << problem << endl;
    return false;
  }
  return true;
}

template<typename T>
void Run(Model<T>* model, string problem, option::Option* options,
    const RandomStreams& streams) {
  ILowerBound<T>* lb_ptr;
  string lb_type;
  if (options[LBTYPE])
    lb_type = options[LBTYPE].arg;
  else {
    if (problem == "tag" || problem == "lasertag") 
      lb_type = "mode";
    else
      lb_type = "random";
  }
  if (!AssignLowerBound(lb_ptr, problem, lb_type, model, streams, options))
    return;

  IUpperBound<T>* ub_ptr;
  AssignUpperBound(ub_ptr, streams, model);
  
  BeliefUpdate<T>* bu_ptr;
  string bu_type = options[BELIEF] ? options[BELIEF].arg : "particle";
  if (!AssignBeliefUpdate(bu_ptr, bu_type, problem, model, streams))
    return;

  VNode<T>::set_model(*model);
  World<T> world(streams.WorldSeed(), *model);
  Solver<T>* solver = 
      new Solver<T>(*model, model->InitialBelief(), *lb_ptr, *ub_ptr, *bu_ptr,
                    streams);
  solver->Init();

  cout << "\nSTARTING STATE:\n";
  model->PrintState(model->GetStartState());

  int total_trials = 0, step;
  double reward; uint64_t obs;
  for (step = 0; 
       !solver->Finished() && (Globals::config.sim_len == -1 || 
                               step < Globals::config.sim_len);
       step++) {
    cout << "\nSTEP " << step + 1 << endl;
    int n_trials = 0;
    int act = solver->Search(Globals::config.time_per_move, n_trials);
    total_trials += n_trials;
    world.Step(act, obs, reward);
    solver->UpdateBelief(act, obs);
  }
  cout << "\n";
  cout << "Number of steps = " << step << endl;
  cout << "Discounted return = " << world.DiscountedReturn() << endl;
  cout << "Undiscounted return = " << world.UndiscountedReturn() << endl;
  cerr << "Average # of trials per move = "
       << (step == 0 ? 0 : (double)total_trials / step) << endl;
}

int main(int argc, char* argv[]) {
  argc -= (argc > 0); argv += (argc > 0); // skip program name if present
  option::Stats stats(usage, argc, argv);
  option::Option* options = new option::Option[stats.options_max];
  option::Option* buffer = new option::Option[stats.buffer_max];
  option::Parser parse(usage, argc, argv, options, buffer);

  string problem;
  if (!options[PROBLEM]) {
    option::printUsage(std::cout, usage);
    return 0;
  }
  problem = options[PROBLEM].arg;

  // Optional parameters
  if (options[DEPTH]) Globals::config.search_depth = atoi(options[DEPTH].arg);
  if (options[DISCOUNT]) Globals::config.discount = atof(options[DISCOUNT].arg);
  if (options[SEED]) Globals::config.root_seed = atoi(options[SEED].arg);
  if (options[TIMEOUT]) Globals::config.time_per_move = atof(options[TIMEOUT].arg);
  if (options[NPARTICLES]) Globals::config.n_particles = atoi(options[NPARTICLES].arg);
  if (options[REGPARAM]) Globals::config.pruning_constant = atof(options[REGPARAM].arg);
  if (options[SIMLEN]) Globals::config.sim_len = atoi(options[SIMLEN].arg);
  if (options[APPROX_BOUND]) Globals::config.approximate_ubound = true;

  RandomStreams streams(Globals::config.n_particles, 
                        Globals::config.search_depth, 
                        Globals::config.root_seed);
  int size = options[SIZE] ? atoi(options[SIZE].arg) : 0;
  int number = options[NUMBER] ? atoi(options[NUMBER].arg) : 0;

  if (problem == "tag")
    Run(new Tag(streams.ModelSeed()), "tag", options, streams);
  else if (problem == "lasertag")
    Run(new LaserTag(streams.ModelSeed()), "lasertag", options, streams);
  else if (problem == "rocksample") {
    if (!size) {
      size = 7;
      number = 8;
    }
    Run(new RockSample(size, number, streams.ModelSeed(), streams), 
        "rocksample", options, streams);
  }
  else if (problem == "tiger")
    Run(new Tiger(streams.ModelSeed()), "tiger", options, streams);
  else if (problem == "bridge")
    Run(new Bridge(streams.ModelSeed()), "bridge", options, streams);
  else if (problem == "pocman")
    Run(new Pocman(streams.ModelSeed(), streams), "pocman", options, streams);
  else
    cout << "Problem must be one of tag, lasertag, rocksample, tiger, bridge "
            "and pocman.\n";

  return 0;
}
