const { Router } = require('express');
const authRoutes = require('./auth.routes');
const quizRoutes = require('./quiz.routes');
const router = Router();

router.use('/auth', authRoutes);
router.use('/quiz', quizRoutes);

module.exports = router;