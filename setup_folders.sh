#!/bin/bash

# ================================================================
# Claude Innovation Workflow Setup Script
# ================================================================
# This script sets up the complete innovation validation workflow
# for Claude Code with persistent agents and markdown-based commands
# ================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Header
echo -e "${PURPLE}================================================================${NC}"
echo -e "${PURPLE}   Claude Innovation Workflow Setup${NC}"
echo -e "${PURPLE}================================================================${NC}\n"

# Function to create file with content
create_file() {
    local filepath=$1
    local content=$2
    mkdir -p $(dirname "$filepath")
    echo "$content" > "$filepath"
    echo -e "${GREEN}✓${NC} Created: $filepath"
}

# Function to show progress
show_progress() {
    echo -e "\n${BLUE}→ $1${NC}"
}

# ================================================================
# STEP 1: Create Directory Structure
# ================================================================
show_progress "Creating directory structure..."

directories=(
    ".claude/commands"
    ".claude/agents/research"
    ".claude/agents/planning"
    ".claude/agents/prompts"
    ".claude/sessions"
    "projects"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo -e "${GREEN}✓${NC} Created directory: $dir"
done
