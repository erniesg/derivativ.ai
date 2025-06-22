const jwt = require('jsonwebtoken');
const supabase = require('../config/supabase');

const supabaseAuthMiddleware = async (req, res, next) => {
  try {
    const jwtToken = req.headers['authorization']?.replace('Bearer ', '');

    if (!jwtToken) {
      return res.status(400).send({ message: 'No JWT token provided' });
    }

    const decoded = jwt.verify(jwtToken, process.env.JWT_SECRET);

    if (!decoded) {
      return res.status(401).send({ message: 'Invalid JWT token' });
    }

    const {
      data: { user },
    } = await supabase.auth.getUser(jwtToken);

    if (!user) {
      return res.status(401).send({ message: 'Invalid JWT token' });
    }

    if (Object.hasOwn(user, 'banned_until')) {
      return res
        .status(401)
        .send({ message: `User is banned until ${user.banned_until}` });
    }

    req.user = user;
    req.jwt = jwtToken;
    next();
  } catch (error) {
    console.error(error);
    return res.status(500).send({ message: 'Something went wrong!' });
  }
}

module.exports = supabaseAuthMiddleware;