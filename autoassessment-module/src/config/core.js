// Configuration constants
const DEFAULT_DIFFICULTY_DISTRIBUTION = {
  easy: 0.5,
  medium: 0.3,
  hard: 0.2,
};

const WMA_WEIGHT = 0.3; // Weight for new performance in WMA calculation
const PADDING_FACTOR = 0.8; // Factor to pad initial grades to avoid false achievement

module.exports = {
  DEFAULT_DIFFICULTY_DISTRIBUTION,
  WMA_WEIGHT,
  PADDING_FACTOR,
};