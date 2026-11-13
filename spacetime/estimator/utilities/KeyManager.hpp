#pragma once 

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/FixedLagSmoother.h>

/// Allocates per-character gtsam::Symbol keys and records each key's timestamp for the fixed-lag smoother.
class KeyManager {
public:
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
  std::map<unsigned char, size_t> counters_;
  KeyTimestampMap allTimestamps_;
  KeyTimestampMap pendingTimestamps_;
};