"""
Setup configuration for Derivativ AI platform.
Enables proper package imports and development installation.
"""

from setuptools import find_packages, setup

setup(
    name="derivativ",
    version="0.1.0",
    description="AI-Powered Math Tutor with Multi-Agent Question Generation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0",
        "jinja2>=3.0.0",
        "openai>=1.0.0",
        "anthropic>=0.50.0",
        "google-generativeai>=0.8.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
        ]
    },
    entry_points={
        "console_scripts": [
            "derivativ-test-apis=scripts.test_live_apis:main",
            "derivativ-setup-keys=scripts.setup_api_keys:main",
        ]
    },
)
