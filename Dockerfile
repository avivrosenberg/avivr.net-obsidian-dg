# Use Node.js as the base image
FROM node:20

# Set the working directory inside the container
WORKDIR /app

# Copy only package.json and package-lock.json for dependency installation
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all project files to the container
COPY . .

# Expose port 8080 for the Eleventy development server
EXPOSE 8080

# Default command is overridden by docker-compose, so no CMD here
