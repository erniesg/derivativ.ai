const express = require('express');
const supabase = require('../config/supabase');

const createUser = async (req, res) => {
  try {
    const {
      data: { user},
    } =  await supabase.auth.getUser(req.jwt)

    if (!user) {
      return res.status(401).send({ message: 'Something went wrong!'})
    }    

    // check if user already exists
  } catch (error) {
    console.log('[create user] error', error)
    return res.status(500).send({ message: 'Something went wrong!'})
  }
}

module.exports = {
  createUser
}