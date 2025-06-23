const { Router } = require('express');
const { auth: authController } = require('../controllers');
const { supabaseAuthMiddleware } = require('../middleware/auth.middleware');
const router = Router();

// Apply authentication middleware to all auth routes
router.use(supabaseAuthMiddleware);

// Profile management routes
router.post('/profile', authController.createProfile);
router.get('/profile', authController.getProfile);
router.put('/profile', authController.updateProfile);

// Backward compatibility
router.post('/', authController.createUser);

module.exports = router;