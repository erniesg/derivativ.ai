const supabase = require("../config/supabase");
const quizService = require("../services/quiz.service");
const Joi = require("joi");

// Validation schema for profile creation
const createProfileSchema = Joi.object({
  school: Joi.string().max(100).optional(),
  role: Joi.string().valid("student", "teacher").default("student"),
});

/**
 * Create or update user profile after Supabase authentication
 */
const createProfile = async (req, res) => {
  try {
    const { error: validationError, value } = createProfileSchema.validate(
      req.body
    );
    if (validationError) {
      return res.status(400).json({
        message: "Validation error",
        details: validationError.details,
      });
    }

    const {
      data: { user },
      error: getUserError,
    } = await supabase.auth.getUser(req.jwt);

    if (getUserError || !user) {
      return res.status(401).json({
        message: "Invalid authentication token",
      });
    }

    const { school, role } = value;

    // Check if profile already exists
    const { data: existingProfile } = await supabase
      .from("profiles")
      .select("*")
      .eq("id", user.id)
      .single();

    if (existingProfile) {
      return res.status(200).json({
        message: "Profile already exists",
        data: existingProfile,
      });
    }

    // Create new profile
    const { data: profile, error: profileError } = await supabase
      .from("profiles")
      .insert({
        id: user.id,
        school: school || null,
        role: role || "student",
        is_active: true,
      })
      .select()
      .single();

    if (profileError) {
      console.error("Error creating profile:", profileError);
      return res.status(500).json({
        message: "Failed to create user profile",
      });
    }

    // Initialize quiz performance for students
    if (role === "student") {
      try {
        await quizService.initializeUserPerformance(user.id);
      } catch (err) {
        console.error("Error initializing user performance:", err);
      }
    }

    res.status(201).json({
      message: "Profile created successfully",
      data: {
        id: profile.id,
        school: profile.school,
        role: profile.role,
        is_active: profile.is_active,
        email: user.email,
      },
    });
  } catch (error) {
    console.error("Error in createProfile:", error);
    return res.status(500).json({
      message: "Internal server error",
    });
  }
};

/**
 * Get current user profile
 */
const getProfile = async (req, res) => {
  try {
    const {
      data: { user },
      error: getUserError,
    } = await supabase.auth.getUser(req.jwt);

    if (getUserError || !user) {
      return res.status(401).json({
        message: "Invalid authentication token",
      });
    }

    const { data: profile, error } = await supabase
      .from("profiles")
      .select("*")
      .eq("id", user.id)
      .single();

    if (error) {
      return res.status(404).json({
        message: "Profile not found",
      });
    }

    res.status(200).json({
      message: "Profile retrieved successfully",
      data: {
        ...profile,
        email: user.email,
      },
    });
  } catch (error) {
    console.error("Error in getProfile:", error);
    return res.status(500).json({
      message: "Internal server error",
    });
  }
};

/**
 * Update user profile
 */
const updateProfile = async (req, res) => {
  try {
    const updateSchema = Joi.object({
      school: Joi.string().max(100).optional(),
    });

    const { error: validationError, value } = updateSchema.validate(req.body);
    if (validationError) {
      return res.status(400).json({
        message: "Validation error",
        details: validationError.details,
      });
    }

    const {
      data: { user },
      error: getUserError,
    } = await supabase.auth.getUser(req.jwt);

    if (getUserError || !user) {
      return res.status(401).json({
        message: "Invalid authentication token",
      });
    }

    const { data: profile, error } = await supabase
      .from("profiles")
      .update(value)
      .eq("id", user.id)
      .select()
      .single();

    if (error) {
      return res.status(500).json({
        message: "Failed to update profile",
      });
    }

    res.status(200).json({
      message: "Profile updated successfully",
      data: {
        ...profile,
        email: user.email,
      },
    });
  } catch (error) {
    console.error("Error in updateProfile:", error);
    return res.status(500).json({
      message: "Internal server error",
    });
  }
};

module.exports = {
  createProfile,
  getProfile,
  updateProfile,
};
