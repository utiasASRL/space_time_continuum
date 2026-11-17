#pragma once 

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/FixedLagSmoother.h>

/**
 * @file KeyManager.hpp
 * @brief Key allocation and timestamp bookkeeping shared by every Types::Model.
 */

/**
 * @brief Allocates per-character gtsam::Symbol keys and records each key's timestamp for
 *        the fixed-lag smoother, which marginalizes a key once its timestamp falls out of
 *        the lag window. Every key a model adds in Types::Model::extendState must come from
 *        here (or at least be stamped here) so that the smoother knows when to drop it.
 */
class KeyManager {
public:
  /// Key -> timestamp (the time of the state the key belongs to), as the smoother takes it.
  using KeyTimestampMap = gtsam::FixedLagSmoother::KeyTimestampMap;

  /// A new key Symbol(chr, n), n counting up per character, stamped with `timestamp`.
  gtsam::Key nextKey(unsigned char chr, double timestamp) {
    gtsam::Key key = gtsam::Symbol(chr, counters_[chr]++);
    allTimestamps_[key] = timestamp; // note: this currently grows unbounded. Can prune later if needed
    pendingTimestamps_[key] = timestamp;
    return key;
  }

  /// The timestamps of the keys made since the last call, emptying that set. Called once per Estimator::estimate().
  KeyTimestampMap drainPending() {
    KeyTimestampMap out = std::move(pendingTimestamps_);
    pendingTimestamps_.clear();
    return out;
  }

  /// The timestamp `key` was made with. @throws std::out_of_range for a key not made here.
  double timestampOf(gtsam::Key key) const { return allTimestamps_.at(key); }

private:
  std::map<unsigned char, size_t> counters_; ///< Next index per Symbol character.
  KeyTimestampMap allTimestamps_;            ///< Every key ever made (never pruned).
  KeyTimestampMap pendingTimestamps_;        ///< Keys made since the last drainPending().
};