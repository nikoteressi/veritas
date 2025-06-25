-- Initialize Veritas database
-- This script is run when the PostgreSQL container starts

-- Create database if it doesn't exist (handled by Docker environment)
-- CREATE DATABASE veritas_db;

-- Create user if it doesn't exist (handled by Docker environment)
-- CREATE USER veritas_user WITH PASSWORD 'veritas_password';
-- GRANT ALL PRIVILEGES ON DATABASE veritas_db TO veritas_user;

-- Connect to the database
\c veritas_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance (will be created by SQLAlchemy, but good to have)
-- These will be created automatically by the application

-- Insert some sample data for testing (optional)
-- This will be handled by the application's CRUD operations

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO veritas_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO veritas_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO veritas_user;
