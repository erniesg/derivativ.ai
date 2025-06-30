const supabase = require("../config/supabase");
const { DEFAULT_DIFFICULTY_DISTRIBUTION, WMA_WEIGHT, PADDING_FACTOR } = require("../config/core");

/**
 * Initialize a new quiz session for a user
 */
const initializeQuizSession = async (userId, totalQuestions = 10) => {
  try {
    const difficultyDistribution = await calculateDynamicDifficulty(userId);

    // Select questions for this session
    const questions = await selectQuestionsForSession(
      difficultyDistribution,
      totalQuestions
    );

      // Create new quiz session
      const { data: session, error } = await supabase
      .from("quiz_sessions")
      .insert({
        user_id: userId,
        difficulty_distribution: difficultyDistribution,
        total_questions: totalQuestions,
        questions_selected: questions.map((q) => q.id),
        status: "active",
      })
      .select()
      .single();

    if (error) throw error;

    return {
      session,
      questions: questions.map((q) => ({
        id: q.id,
        topic_id: q.topic_id,
        difficulty: q.difficulty,
        stem: q.stem,
        options: q.options,
      })),
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
      .select("topic_id, wma_grade, total_attempts")
      .eq("user_id", userId);

    if (error) throw error;

    const totalAttempts = performance.reduce((sum, p) => sum + p.total_attempts, 0); // if no attempts, return default distribution

    if (!performance || performance.length === 0 || totalAttempts === 0) {
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

    const easyQuestions = await fetchQuestionsByDifficulty(Object.keys(questionsPerDifficulty)[0], questionsPerDifficulty.easy);
    const mediumQuestions = await fetchQuestionsByDifficulty(Object.keys(questionsPerDifficulty)[1], questionsPerDifficulty.medium);
    const hardQuestions = await fetchQuestionsByDifficulty(Object.keys(questionsPerDifficulty)[2], questionsPerDifficulty.hard);
    
    selectedQuestions.push(...easyQuestions.slice(0, questionsPerDifficulty.easy));
    selectedQuestions.push(...mediumQuestions.slice(0, questionsPerDifficulty.medium));
    selectedQuestions.push(...hardQuestions.slice(0, questionsPerDifficulty.hard));
  
    return durstenfeldShuffle(selectedQuestions);

  } catch (error) {
    console.error("Error selecting questions:", error);
    throw error;
  }
};

/**
 * Submit a quiz response
 */
const submitQuizResponse = async (userId,sessionId, questionId, userAnswer) => {
  try {
    // Check if this question has already been answered in this session
    const { data: existingResponse, error: checkError } = await supabase
      .from("quiz_responses")
      .select("id")
      .eq("quiz_session_id", sessionId)
      .eq("question_id", questionId)
      .single();

    if (checkError && checkError.code !== 'PGRST116') {
      // PGRST116 is "not found" error, which is expected if no response exists
      throw checkError;
    }

    if (existingResponse) {
      throw new Error("Question has already been answered in this session");
    }

    const {data: session, error: sessionError} = await supabase
      .from("quiz_sessions")
      .select('*')
      .eq('id', sessionId)
      .eq('user_id', userId)
      .single();

    if (sessionError) throw sessionError;

    const selectedQuestion = session.questions_selected.includes(questionId);
  
    if (!selectedQuestion) {
      throw new Error("Invalid question");
    }

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
const completeQuizSession = async (sessionId, userId) => {
  try {
    // Get session details
    const { data: session, error: sessionError } = await supabase
      .from("quiz_sessions")
      .select("*")
      .eq("id", sessionId)
      .eq("user_id", userId)
      .eq("status", "active")
      .single();

    if (sessionError) throw sessionError;


    const { data: questions, error: questionsError} = await supabase
      .from("questions")
      .select("topic_id")
      .in("id", session.questions_selected);

    if (questionsError) throw questionsError;

    const topicIds = questions.map(q => q.topic_id);
    const topicCounts = topicIds.reduce((acc, topicId) => {
      acc[topicId] = (acc[topicId] || 0) + 1;
      return acc; 
    }, {});

    // Get all responses for this session
    const { data: responses, error: responsesError } = await supabase
      .from("quiz_responses")
      .select(
        `
        *,
        questions:question_id(*)
      `
      )
      .eq("quiz_session_id", sessionId);

    if (responsesError) throw responsesError;

    // Calculate performance by topic
    const topicPerformance = {};

    for (const [topicId, count] of Object.entries(topicCounts)) {
      topicPerformance[topicId] = { correct: 0, total: count };
    }

    responses.forEach((response) => {
      const topicId = response.questions.topic_id;
      if (response.is_correct) {
        topicPerformance[topicId].correct++;
      }
    })

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

const fetchQuestionsByDifficulty = async (difficulty, count) => {
  try {
    const { data: questions, error } = await supabase
    .from("questions")
    .select("*")
    .eq("difficulty", difficulty)
    .eq("status", "approved")
    .limit(count * 2) // Fetch more for better randomization
    .order("created_at", { ascending: false });

    if (error) throw error;
    return questions;
  } catch (error) {
    console.error("Error fetching questions by difficulty:", error);
    throw error;
  }
}

const durstenfeldShuffle = (array) => {
  for (let i = array.length - 1; i> 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));

    [array[i], array[j]] = [array[j], array[i]];
  }

  return array;
}

const getTopicIdFromQuestionId = async (questionId) => {
  const { data: question, error } = await supabase
    .from("questions")
    .select("topic_id")
    .eq("id", questionId)
    .single();

  if (error) throw error;

  return question.topic_id;
}

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
