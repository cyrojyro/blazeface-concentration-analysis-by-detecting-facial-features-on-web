module.exports = {
  env: {
    browser: true,
    es2020: true,
    amd: true,
  },
  extends: ["eslint:recommended"],
  parserOptions: {
    ecmaVersion: 11,
    sourceType: "module",
  },
  rules: {},
  ignorePatterns: ["main.js"],
};
