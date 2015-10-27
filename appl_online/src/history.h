#ifndef HISTORY_H
#define HISTORY_H

#include <vector>

// Encapsulates a history of actions and observations.
class History {
 public:
  History() : hash_(1, 0) {}

  void Add(int action, uint64_t obs) {
    actions_.push_back(action);
    observations_.push_back(obs);

    size_t next_hash = hash_.back();
    Globals::hash_combine(next_hash, action);
    Globals::hash_combine(next_hash, obs);
    hash_.push_back(next_hash);
  }

  void RemoveLast() { 
    actions_.pop_back();
    observations_.pop_back();
    hash_.pop_back();
  }

  int Action(int t) const { return actions_[t]; }

  uint64_t Observation(int t) const { return observations_[t]; }

  size_t Size() const { return actions_.size(); }

  void Truncate(int d) {
    actions_.resize(d);
    observations_.resize(d);
    hash_.resize(d + 1);
  }

  int LastAction() const { return actions_.back(); }

  uint64_t LastObservation() const { return observations_.back(); }

  // Hash of the current history. 
  // Useful to uniquely identify the history quickly.
  size_t Hash() const { return hash_.back(); }

 private:
  std::vector<int> actions_;
  std::vector<uint64_t> observations_;
  // Hashes of prefixes of the history. Since we use a rolling hash, computing
  // the new hash when the history gets updated is fast.
  std::vector<size_t> hash_;
};

#endif
