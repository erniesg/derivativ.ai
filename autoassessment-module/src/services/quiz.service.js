const supabase = require("../config/supabase");

// Configuration constants
const DEFAULT_DIFFICULTY_DISTRIBUTION = {
  easy: 0.5,
  medium: 0.3,
  hard: 0.2,
};

const WMA_WEIGHT = 0.3; // Weight for new performance in WMA calculation
const PADDING_FACTOR = 0.8; // Factor to pad initial grades to avoid false achievement

/**
 * Initialize a new quiz session for a user
 */
const initializeQuizSession = async (userId, totalQuestions = 10) => {
  try {
    await initializeUserPerformance(userId);
    const difficultyDistribution = await calculateDynamicDifficulty(userId);

    // Create new quiz session
    const { data: session, error } = await supabase
      .from("quiz_sessions")
      .insert({
        user_id: userId,
        difficulty_distribution: difficultyDistribution,
        total_questions: totalQuestions,
        status: "active",
      })
      .select()
      .single();

    if (error) throw error;

    // Select questions for this session
    const questions = await selectQuestionsForSession(
      session.id,
      difficultyDistribution,
      totalQuestions
    );

    // Update session with selected questions
    await supabase
      .from("quiz_sessions")
      .update({ questions_selected: questions.map((q) => q.id) })
      .eq("id", session.id);

    return {
      session,
      questions: questions.map((q) => ({
        id: q.id,
        topic_id: q.topic_id,
        difficulty: q.difficulty,
        stem: q.stem,
        options: q.options,
      })), // Remove correct answers from response
    };
  } catch (error) {
    console.error("Error initializing quiz session:", error);
    throw error;
  }
};

/**
 * Calculate dynamic difficulty distribution based on user's WMA scores
 */
const calculateDynamicDifficulty = async (userId) => {
  try {
    // Get user's current WMA scores for all topics
    const { data: performance, error } = await supabase
      .from("student_topic_performance")
      .select("topic_id, wma_grade")
      .eq("user_id", userId);

    if (error) throw error;

    if (!performance || performance.length === 0) {
      return DEFAULT_DIFFICULTY_DISTRIBUTION;
    }

    // Calculate average WMA across all topics
    const avgWma =
      performance.reduce((sum, p) => sum + parseFloat(p.wma_grade), 0) /
      performance.length;

    // Apply softmax transformation to determine difficulty distribution
    return applySoftmaxDifficulty(avgWma);
  } catch (error) {
    console.error("Error calculating dynamic difficulty:", error);
    return DEFAULT_DIFFICULTY_DISTRIBUTION;
  }
};

/**
 * Apply softmax-based difficulty adjustment
 */
const applySoftmaxDifficulty = (avgWma) => {
  // Normalize WMA score to 0-1 range (1-9 scale)
  const normalizedScore = (avgWma - 1) / 8;

  // Define difficulty logits based on performance
  let easyLogit, mediumLogit, hardLogit;

  if (normalizedScore < 0.3) {
    // Low performance: more easy questions
    easyLogit = 2.0;
    mediumLogit = 0.5;
    hardLogit = -1.0;
  } else if (normalizedScore < 0.7) {
    // Medium performance: balanced distribution
    easyLogit = 1.0;
    mediumLogit = 1.2;
    hardLogit = 0.8;
  } else {
    // High performance: more challenging questions
    easyLogit = 0.0;
    mediumLogit = 1.5;
    hardLogit = 2.0;
  }

  // Apply softmax
  const expEasy = Math.exp(easyLogit);
  const expMedium = Math.exp(mediumLogit);
  const expHard = Math.exp(hardLogit);
  const sumExp = expEasy + expMedium + expHard;

  return {
    easy: expEasy / sumExp,
    medium: expMedium / sumExp,
    hard: expHard / sumExp,
  };
};

/**
 * Select questions for a quiz session based on difficulty distribution
 */
const selectQuestionsForSession = async (
  sessionId,
  distribution,
  totalQuestions
) => {
  try {
    const questionsPerDifficulty = {
      easy: Math.round(totalQuestions * distribution.easy),
      medium: Math.round(totalQuestions * distribution.medium),
      hard: Math.round(totalQuestions * distribution.hard),
    };

    // Adjust for rounding errors
    const totalAllocated =
      questionsPerDifficulty.easy +
      questionsPerDifficulty.medium +
      questionsPerDifficulty.hard;
    if (totalAllocated !== totalQuestions) {
      questionsPerDifficulty.easy += totalQuestions - totalAllocated;
    }

    const selectedQuestions = [];

    // Select questions for each difficulty level
    for (const [difficulty, count] of Object.entries(questionsPerDifficulty)) {
      if (count > 0) {
        const { data: questions, error } = await supabase
          .from("questions")
          .select("*")
          .eq("difficulty", difficulty)
          .eq("status", "approved")
          .limit(count * 2) // Get more questions than needed for randomization
          .order("created_at", { ascending: false });

        if (error) throw error;

        // Randomly select the required number of questions
        const shuffled = questions.sort(() => 0.5 - Math.random());
        selectedQuestions.push(...shuffled.slice(0, count));
      }
    }

    // Shuffle all selected questions
    return selectedQuestions.sort(() => 0.5 - Math.random());
  } catch (error) {
    console.error("Error selecting questions:", error);
    throw error;
  }
};

/**
 * Submit a quiz response
 */
const submitQuizResponse = async (sessionId, questionId, userAnswer) => {
  try {
    const { data: question, error: questionError } = await supabase
      .from("questions")
      .select("correct_answer, topic_id")
      .eq("id", questionId)
      .single();

    if (questionError) throw questionError;

    const isCorrect =
      userAnswer.toLowerCase().trim() ===
      question.correct_answer.toLowerCase().trim();

    // Insert quiz response
    const { data: response, error } = await supabase
      .from("quiz_responses")
      .insert({
        quiz_session_id: sessionId,
        question_id: questionId,
        user_answer: userAnswer,
        is_correct: isCorrect,
      })
      .select()
      .single();

    if (error) throw error;

    return {
      ...response,
      correct_answer: question.correct_answer,
      topic_id: question.topic_id,
    };
  } catch (error) {
    console.error("Error submitting quiz response:", error);
    throw error;
  }
};

/**
 * Complete a quiz session and update WMA scores
 */
const completeQuizSession = async (sessionId) => {
  try {
    // Get session details
    const { data: session, error: sessionError } = await supabase
      .from("quiz_sessions")
      .select("*")
      .eq("id", sessionId)
      .single();

    if (sessionError) throw sessionError;

    // Get all responses for this session
    const { data: responses, error: responsesError } = await supabase
      .from("quiz_responses")
      .select(
        `
        *,
        questions(topic_id)
      `
      )
      .eq("quiz_session_id", sessionId);

    if (responsesError) throw responsesError;

    // Calculate performance by topic
    const topicPerformance = {};
    responses.forEach((response) => {
      const topicId = response.questions.topic_id;
      if (!topicPerformance[topicId]) {
        topicPerformance[topicId] = { correct: 0, total: 0 };
      }
      topicPerformance[topicId].total++;
      if (response.is_correct) {
        topicPerformance[topicId].correct++;
      }
    });

    // Update WMA for each topic
    const topicScores = {};
    for (const [topicId, performance] of Object.entries(topicPerformance)) {
      const score = (performance.correct / performance.total) * 9; // Convert to 1-9 scale
      topicScores[topicId] = score;
      await updateWMA(session.user_id, parseInt(topicId), score, sessionId);
    }

    // Calculate overall performance
    const totalCorrect = responses.filter((r) => r.is_correct).length;
    const overallScore = (totalCorrect / responses.length) * 100;

    // Create quiz result
    const { data: result, error: resultError } = await supabase
      .from("quiz_results")
      .insert({
        quiz_session_id: sessionId,
        user_id: session.user_id,
        topic_scores: topicScores,
        overall_score: overallScore,
        total_questions: responses.length,
        correct_answers: totalCorrect,
        completion_time_seconds: Math.floor(
          (new Date() - new Date(session.started_at)) / 1000
        ),
      })
      .select()
      .single();

    if (resultError) throw resultError;

    // Update session status
    await supabase
      .from("quiz_sessions")
      .update({
        status: "completed",
        completed_at: new Date().toISOString(),
      })
      .eq("id", sessionId);

    return result;
  } catch (error) {
    console.error("Error completing quiz session:", error);
    throw error;
  }
};

/**
 * Update Weighted Moving Average for a topic
 */
const updateWMA = async (userId, topicId, newScore, sessionId) => {
  try {
    // Get current performance record
    const { data: currentPerformance, error: fetchError } = await supabase
      .from("student_topic_performance")
      .select("*")
      .eq("user_id", userId)
      .eq("topic_id", topicId)
      .single();

    if (fetchError) throw fetchError;

    const previousWma = parseFloat(currentPerformance.wma_grade);

    // Calculate new WMA: WMA = (1-α) × previous_WMA + α × new_score
    const newWma = (1 - WMA_WEIGHT) * previousWma + WMA_WEIGHT * newScore;

    // Ensure WMA stays within bounds
    const boundedWma = Math.max(1, Math.min(9, newWma));

    // Update performance record
    await supabase
      .from("student_topic_performance")
      .update({
        current_grade: newScore,
        wma_grade: boundedWma,
        total_attempts: currentPerformance.total_attempts + 1,
        last_updated: new Date().toISOString(),
      })
      .eq("user_id", userId)
      .eq("topic_id", topicId);

    // Record WMA history
    await supabase.from("wma_history").insert({
      user_id: userId,
      topic_id: topicId,
      quiz_session_id: sessionId,
      previous_wma: previousWma,
      new_wma: boundedWma,
      performance_score: newScore,
      weight_factor: WMA_WEIGHT,
    });

    return boundedWma;
  } catch (error) {
    console.error("Error updating WMA:", error);
    throw error;
  }
};

/**
 * Initialize user performance records with padding
 */
const initializeUserPerformance = async (userId) => {
  try {
    // Get all topics
    const { data: topics, error: topicsError } = await supabase
      .from("topics")
      .select("id");

    if (topicsError) throw topicsError;

    // Apply padding to initial grades to avoid false sense of achievement
    const paddedGrade = 4.0 * PADDING_FACTOR;

    // Create performance records for each topic
    const performanceRecords = topics.map(topic => ({
      user_id: userId,
      topic_id: topic.id,
      current_grade: paddedGrade,
      wma_grade: paddedGrade,
      total_attempts: 0
    }));

    // Insert all records at once, ignore conflicts (user already has records)
    const { error: insertError } = await supabase
      .from("student_topic_performance")
      .upsert(performanceRecords, { 
        onConflict: 'user_id,topic_id',
        ignoreDuplicates: true 
      });

    if (insertError) throw insertError;

  } catch (error) {
    console.error("Error initializing user performance:", error);
    throw error;
  }
};

/**
 * Get user's current performance across all topics
 */
const getUserPerformance = async (userId) => {
  try {
    const { data: performance, error } = await supabase
      .from("student_topic_performance")
      .select(
        `
        *,
        topics(topic_name, description)
      `
      )
      .eq("user_id", userId);

    if (error) throw error;

    return performance;
  } catch (error) {
    console.error("Error getting user performance:", error);
    throw error;
  }
};

/**
 * Get user's quiz history
 */
const getQuizHistory = async (userId, limit = 10) => {
  try {
    const { data: history, error } = await supabase
      .from("quiz_results")
      .select(
        `
        *,
        quiz_sessions(started_at, completed_at, difficulty_distribution)
      `
      )
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(limit);

    if (error) throw error;

    return history;
  } catch (error) {
    console.error("Error getting quiz history:", error);
    throw error;
  }
};

module.exports = {
  initializeQuizSession,
  calculateDynamicDifficulty,
  applySoftmaxDifficulty,
  selectQuestionsForSession,
  submitQuizResponse,
  completeQuizSession,
  updateWMA,
  initializeUserPerformance,
  getUserPerformance,
  getQuizHistory,
};
