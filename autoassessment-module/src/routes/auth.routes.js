const { Router } = require('express');
const { authController } = require('../controllers');
const { authMiddleware } = require('../middleware');
const { createUser} = authController
const router = Router();

router.use(authMiddleware);
router.post('/', createUser);

module.exports = router;