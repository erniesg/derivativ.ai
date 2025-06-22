const { Router } = require("express");
const quizController = require("../controllers/quiz.controller");
const authMiddleware = require("../middleware/auth.middleware");

const {
  initializeQuiz,
  getQuizSession,
  submitResponse,
  completeQuiz,
  getUserPerformance,
  getQuizHistory,
  getAnalytics,
  getDifficultyDistribution,
  getTopics,
} = quizController;

const router = Router();

// Apply authentication middleware to all quiz routes
router.use(authMiddleware.verifyToken);

// Quiz session management
router.post("/initialize", initializeQuiz);
router.get("/session/:sessionId", getQuizSession);
router.post("/submit-response", submitResponse);
router.post("/complete", completeQuiz);

// User performance and analytics
router.get("/performance", getUserPerformance);
router.get("/history", getQuizHistory);
router.get("/analytics", getAnalytics);

// Quiz configuration
router.get("/difficulty-distribution", getDifficultyDistribution);
router.get("/topics", getTopics);

module.exports = router;
