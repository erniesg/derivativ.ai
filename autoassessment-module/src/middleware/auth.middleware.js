const jwt = require("jsonwebtoken");
const supabase = require("../config/supabase");

/**
 * Main authentication middleware
 */
const supabaseAuthMiddleware = async (req, res, next) => {
  try {
    const jwtToken = req.headers["authorization"]?.replace("Bearer ", "");
    
    const {
      data: { user },
    } = await supabase.auth.getUser(jwtToken);

    if (!user) {
      return res.status(401).send({ message: "Invalid JWT token" });
    }

    if (Object.hasOwn(user, "banned_until")) {
      return res
        .status(401)
        .send({ message: `User is banned until ${user.banned_until}` });
    }

    req.user = user;
    req.jwt = jwtToken;
    next();
  } catch (error) {
    console.error(error);
    return res.status(500).send({ message: "Something went wrong!" });
  }
};

/**
 * Enhanced authentication middleware for quiz system
 */
const verifyToken = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return res.status(401).json({
        message: "Authorization token required",
      });
    }

    const token = authHeader.split(" ")[1];

    // Verify token with Supabase
    const {
      data: { user },
      error,
    } = await supabase.auth.getUser(token);

    if (error || !user) {
      return res.status(401).json({
        message: "Invalid or expired token",
      });
    }

    // Get user profile
    const { data: profile, error: profileError } = await supabase
      .from("profiles")
      .select("*")
      .eq("id", user.id)
      .single();

    if (profileError && profileError.code !== "PGRST116") {
      // PGRST116 = no rows returned
      console.error("Error fetching user profile:", profileError);
      return res.status(500).json({
        message: "Error fetching user profile",
      });
    }

    // Attach user information to request
    req.user = {
      id: user.id,
      email: user.email,
      role: profile?.role || "student",
      school: profile?.school,
      isActive: profile?.is_active || false,
    };

    next();
  } catch (error) {
    console.error("Auth middleware error:", error);
    return res.status(401).json({
      message: "Authentication failed",
    });
  }
};

/**
 * Check if user has required role
 */
const requireRole = (requiredRole) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        message: "Authentication required",
      });
    }

    if (req.user.role !== requiredRole && req.user.role !== "superadmin") {
      return res.status(403).json({
        message: `${requiredRole} role required`,
      });
    }

    next();
  };
};

/**
 * Check if user is active
 */
const requireActiveUser = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({
      message: "Authentication required",
    });
  }

  if (!req.user.isActive) {
    return res.status(403).json({
      message: "Account is not active. Please contact your administrator.",
    });
  }

  next();
};

module.exports = {
  supabaseAuthMiddleware,
  verifyToken,
  requireRole,
  requireActiveUser,
};
