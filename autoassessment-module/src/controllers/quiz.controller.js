const quizService = require("../services/quiz.service");
const Joi = require("joi");

// Validation schemas
const initializeQuizSchema = Joi.object({
  totalQuestions: Joi.number().integer().min(10).default(10),
});

const submitResponseSchema = Joi.object({
  sessionId: Joi.string().uuid().required(),
  questionId: Joi.string().uuid().required(),
  answer: Joi.string().required(),
});

const completeQuizSchema = Joi.object({
  sessionId: Joi.string().uuid().required(),
});

/**
 * Initialize a new quiz session
 */
const initializeQuiz = async (req, res) => {
  try {
    const { error, value } = initializeQuizSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        message: "Validation error",
        details: error.details,
      });
    }

    const userId = req.user.id;
    const { totalQuestions } = value;

    const result = await quizService.initializeQuizSession(
      userId,
      totalQuestions
    );

    res.status(200).json({
      message: "Quiz session initialized successfully",
      data: result,
    });
  } catch (error) {
    console.error("Error in initializeQuiz:", error);
    res.status(500).json({
      message: "Failed to initialize quiz session",
      error: error.message,
    });
  }
};

/**
 * Submit a quiz response
 */
const submitResponse = async (req, res) => {
  try {
    const { error, value } = submitResponseSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        error: error.details,
      });
    }
    
    const user = req.user;

    const { sessionId, questionId, answer } = value;

    const result = await quizService.submitQuizResponse(
      user.id,
      sessionId,
      questionId,
      answer
    );

    res.status(200).json({
      isCorrect: result.is_correct,
      correctAnswer: result.correct_answer,
      topicId: result.topic_id,
    });
  } catch (error) {
    console.error("Error in submitResponse:", error);
    
    // Handle duplicate response error specifically
    if (error.message === "Question has already been answered in this session") {
      return res.status(409).json({
        error: "Question has already been answered in this session",
        code: "DUPLICATE_RESPONSE"
      });
    }
    
    res.status(500).json({
      error: error.message,
    });
  }
};

const completeQuiz = async (req, res) => {
  try {
    const { error, value } = completeQuizSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        details: error.details,
      });
    }

    const { sessionId } = value;
    const user = req.user;

    const result = await quizService.completeQuizSession(sessionId, user.id);

    res.status(200).json({
      data: result,
    });
  } catch (error) {
    console.error("Error in completeQuiz:", error);
    res.status(500).json({
      error: error.message,
    });
  }
};

/**
 * Get user's performance across all topics
 */
const getUserPerformance = async (req, res) => {
  try {
    const userId = req.user.id;

    const performance = await quizService.getUserPerformance(userId);

    performance.forEach((p) => {
      if (p.total_attempts == 0 && p.current_grade == 3.2) {
        p.current_grade = 0;
        p.wma_grade = 0;
      }
    });
    
    res.status(200).json({
      data: performance,
    });
    
  } catch (error) {
    console.error("Error in getUserPerformance:", error);
    res.status(500).json({
      error: error.message,
    });
  }
};

/**
 * Get user's quiz history
 */
const getQuizHistory = async (req, res) => {
  try {
    const userId = req.user.id;
    const limit = parseInt(req.query.limit) || 10;

    const history = await quizService.getQuizHistory(userId, limit);

    res.status(200).json({
      data: history,
    });
  } catch (error) {
    console.error("Error in getQuizHistory:", error);
    res.status(500).json({
      message: "Failed to retrieve quiz history",
    });
  }
};

/**
 * Get current difficulty distribution for user
 */
const getDifficultyDistribution = async (req, res) => {
  try {
    const userId = req.user.id;

    const distribution = await quizService.calculateDynamicDifficulty(userId);

    res.status(200).json({
      message: "Difficulty distribution retrieved successfully",
      data: {
        distribution,
        explanation: {
          easy: `${Math.round(distribution.easy * 100)}% easy questions`,
          medium: `${Math.round(distribution.medium * 100)}% medium questions`,
          hard: `${Math.round(distribution.hard * 100)}% hard questions`,
        },
      },
    });
  } catch (error) {
    console.error("Error in getDifficultyDistribution:", error);
    res.status(500).json({
      message: "Failed to retrieve difficulty distribution",
      error: error.message,
    });
  }
};

/**
 * Get topics list
 */
const getTopics = async (req, res) => {
  try {
    const supabase = require("../config/supabase");

    const { data: topics, error } = await supabase
      .from("topics")
      .select("*")
      .order("id");

    if (error) throw error;

    res.status(200).json({
      data: topics,
    });
  } catch (error) {
    console.error("Error in getTopics:", error);
    res.status(500).json({
      message: "Failed to retrieve topics",
      error: error.message,
    });
  }
};

/**
 * Get quiz session details
 */
const getQuizSession = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const userId = req.user.id;

    const supabase = require("../config/supabase");

    const { data: session, error } = await supabase
      .from("quiz_sessions")
      .select("*")
      .eq("id", sessionId)
      .eq("user_id", userId)
      .eq("status", "active")
      .single();

    if (error) {
      if (error.code === "PGRST116") {
        return res.status(404).json({
          message: "Quiz session not found",
        });
      }

      throw error;
    };

    res.status(200).json({
       data: session,
    });
  } catch (error) {
    console.error("Error in getQuizSession:", error);
    res.status(500).json({
      message: "Failed to retrieve quiz session",
    });
  }
};

/**
 * Get analytics dashboard data
 */
const getAnalytics = async (req, res) => {
  try {
    const userId = req.user.id;

    // Get performance data and history in parallel
    const [performance, history] = await Promise.all([
      quizService.getUserPerformance(userId),
      quizService.getQuizHistory(userId, 5),
    ]);

    // Calculate analytics
    const analytics = {
      totalQuizzes: history.length,
      averageScore:
        history.length > 0
          ? history.reduce((sum, quiz) => sum + quiz.overall_score, 0) /
            history.length
          : 0,
        topicsPerformance: performance.map((p) => ({
        topicName: p.topics.topic_name,
        currentGrade: p.total_attempts > 0 ? p.current_grade : 0,
        wmaGrade: p.total_attempts > 0 ? p.wma_grade : 0,
        totalAttempts: p.total_attempts,
        trend:
          p.wma_grade > p.current_grade
            ? "improving"
            : p.wma_grade < p.current_grade
            ? "declining"
            : "stable",
      })),
      recentQuizzes: history.slice(0, 3),
      weakestTopics: [],
      strongestTopics: [],
    };

    res.status(200).json({
      data: analytics,
    });
  } catch (error) {
    console.error("Error in getAnalytics:", error);
    res.status(500).json({
      message: "Failed to retrieve analytics data",
    });
  }
};

module.exports = {
  initializeQuiz,
  submitResponse,
  completeQuiz,
  getUserPerformance,
  getQuizHistory,
  getDifficultyDistribution,
  getTopics,
  getQuizSession,
  getAnalytics,
};
