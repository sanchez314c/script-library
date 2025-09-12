#!/usr/bin/env python3

####################################################################################
#                                                                                  #
#   ██████╗ ███████╗████████╗    ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗    #
#  ██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝    #
#  ██║  ███╗█████╗     ██║       ███████╗██║ █╗ ██║██║█████╗     ██║     ╚████╔╝     #
#  ██║   ██║██╔══╝     ██║       ╚════██║██║███╗██║██║██╔══╝     ██║      ╚██╔╝      #
#  ╚██████╔╝███████╗   ██║       ███████║╚███╔███╔╝██║██╗         ██║       ██║       #
#   ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝         ╚═╝       ╚═╝       #
#                                                                                  #
####################################################################################
#
# Script Name: github-org-cloner.py
#
# Author: Your Name <your.email@example.com> (Please update this)
#
# Date Created: 2025-05-24
#
# Last Modified: 2025-05-24
#
# Version: 1.0.0
#
# Description: Downloads all public repositories from a specified GitHub
#              organization using the GitHub API and local Git command.
#
# Usage: python github-org-cloner.py
#        (Ensure GH_TOKEN environment variable is set for authentication)
#
# Dependencies: Python 3, requests library, Git command-line tool
#
# GitHub: https://github.com/your_username/your_repo (Please update this)
#
# Notes: This script requires Git to be installed and accessible in the
#        system's PATH. It uses the GH_TOKEN environment variable for GitHub
#        API authentication to avoid rate limiting. Repositories are cloned
#        into a subdirectory named after the organization within './cloned_repos'.
#
####################################################################################

"""
GitHub Organization Repository Cloner
=====================================

This script automates the process of downloading (cloning) all public
repositories from a specified GitHub organization. It leverages the GitHub API
to fetch the list of repositories and then uses the local Git command-line
tool to perform the cloning operations.

For authenticated requests to the GitHub API (which is recommended to avoid
strict rate limits and to access private repositories if the token has
appropriate permissions), the script expects a GitHub Personal Access Token
to be available in an environment variable named `GH_TOKEN`.
"""

import requests
import subprocess
import os
import time
import sys

# --- Configuration ---
ORG_NAME = "openai"  # The GitHub organization to clone from
CLONE_DIR_BASE = "./cloned_repos"  # Base directory to store cloned repositories
DELAY_BETWEEN_CLONES = 5  # Seconds to wait between clone attempts
ITEMS_PER_PAGE = 100  # Request this many items per page from GitHub API (max 100)

# --- Get GitHub Token from Environment Variable ---
GITHUB_TOKEN = os.getenv('GH_TOKEN')
HEADERS = {}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"
    print("Successfully loaded GH_TOKEN from environment variables.")
else:
    print("Warning: GH_TOKEN environment variable not found. Proceeding without authentication.")
    print("You may encounter stricter API rate limits.")

# GitHub API URL for organization repositories
API_BASE_URL = f"https://api.github.com/orgs/{ORG_NAME}/repos"

# --- Create a specific directory for this organization's clones ---
CLONE_DESTINATION_DIR = os.path.join(CLONE_DIR_BASE, ORG_NAME)
if not os.path.exists(CLONE_DESTINATION_DIR):
    try:
        os.makedirs(CLONE_DESTINATION_DIR)
        print(f"Created base directory for clones: {os.path.abspath(CLONE_DESTINATION_DIR)}")
    except OSError as e:
        print(f"Error creating directory {CLONE_DESTINATION_DIR}: {e}. Please check permissions and path.")
        sys.exit(1)
else:
    print(f"Cloning into existing directory: {os.path.abspath(CLONE_DESTINATION_DIR)}")


# --- 1. Function to fetch all repository clone URLs ---
def get_all_repository_clone_urls(api_url, headers):
    all_clone_urls_and_names = [] # Store tuples of (name, clone_url)
    page = 1
    print(f"\n--- Fetching repository list for '{ORG_NAME}' ---")
    while True:
        print(f"Fetching page {page} of repositories...")
        params = {"per_page": ITEMS_PER_PAGE, "page": page, "type": "public"} # Added type=public
        response = None # Initialize response to None
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()  # Check for HTTP errors
        except requests.exceptions.Timeout:
            print("Error: Request to GitHub API timed out. Check your internet connection or try again later.")
            return [] # Return empty list on timeout
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repositories: {e}")
            status_code = getattr(response, 'status_code', None) # Safely get status_code
            if status_code == 401:
                print("Error 401: Unauthorized. Your GH_TOKEN might be invalid or lack necessary permissions.")
            elif status_code == 403:
                print("Error 403: Forbidden. This could be due to rate limiting or insufficient permissions.")
                print("If rate limited, check response headers for 'X-RateLimit-Remaining' and 'X-RateLimit-Reset'.")
            return [] # Return empty list on error

        repos_on_page = response.json()

        if not repos_on_page:
            print("No more repositories found on this page.")
            break

        for repo_info in repos_on_page:
            if repo_info.get('clone_url') and repo_info.get('name'):
                # Exclude forks if you want:
                # if not repo_info.get('fork', False):
                all_clone_urls_and_names.append((repo_info['name'], repo_info['clone_url']))
            else:
                print(f"Warning: Insufficient data (name or clone_url missing) for a repository entry. Skipping.")
        
        if 'next' not in response.links:
            print("Reached the last page of repositories.")
            break
        
        page += 1
        # Brief pause to be respectful to the API, especially if not authenticated
        time.sleep(1 if GITHUB_TOKEN else 2) 

    print(f"Found {len(all_clone_urls_and_names)} public repository clone URLs for '{ORG_NAME}'.")
    return all_clone_urls_and_names

# --- 2. Function to clone a single repository ---
def clone_repository(repo_name, repo_url, destination_folder):
    target_path = os.path.join(destination_folder, repo_name)

    if os.path.exists(target_path):
        print(f"Repository '{repo_name}' already exists in '{target_path}'. Skipping clone.")
        # Optional: Add logic here to 'git pull' if the repo exists
        # For now, just skipping.
        return "skipped"

    print(f"Attempting to clone '{repo_name}' from {repo_url} into '{target_path}'...")
    try:
        # Using HTTPS clone URL.
        process = subprocess.run(
            ["git", "clone", repo_url, target_path], 
            check=False,  # We will check returncode manually
            capture_output=True, 
            text=True,
            timeout=300 # 5 minutes timeout for clone
        )
        if process.returncode == 0:
            print(f"Successfully cloned '{repo_name}'.")
            return "success"
        else:
            print(f"Failed to clone '{repo_name}'. Git command exited with code {process.returncode}.")
            print(f"Git stdout:\n{process.stdout}")
            print(f"Git stderr:\n{process.stderr}")
            return "failed"
    except subprocess.TimeoutExpired:
        print(f"Error: Timeout while cloning '{repo_name}'. It might be too large or network is slow.")
        return "failed"
    except FileNotFoundError:
        print("Error: 'git' command not found. Please ensure Git is installed and in your system's PATH.")
        return "failed" 
    except Exception as e:
        print(f"An unexpected error occurred while cloning '{repo_name}': {e}")
        return "failed"

# --- Main script execution ---
if __name__ == "__main__":
    print(f"--- Starting GitHub Organization Cloner ---")
    print(f"Target Organization: {ORG_NAME}")
    print(f"Cloning into: {os.path.abspath(CLONE_DESTINATION_DIR)}")
    print(f"Delay between clones: {DELAY_BETWEEN_CLONES} seconds")
    if not GITHUB_TOKEN:
        print("Reminder: Running without a GH_TOKEN may lead to API rate limits more quickly.")
    print("-" * 40)

    # Check for Git early
    try:
        git_version_process = subprocess.run(["git", "--version"], capture_output=True, check=True, text=True)
        print(f"Git command found. Version: {git_version_process.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print("Critical Error: 'git' command not found or not executable. Please ensure Git is installed and in your system's PATH.")
        print(f"Details: {e}")
        sys.exit(1)


    # Get the list of repositories
    repos_to_clone = get_all_repository_clone_urls(API_BASE_URL, HEADERS)

    if not repos_to_clone:
        print("No repositories found or failed to fetch list. Exiting.")
        sys.exit(0)

    total_repos = len(repos_to_clone)
    print(f"\n--- Starting download of {total_repos} repositories ---")
    
    successful_clones = 0
    failed_clones = 0
    skipped_clones = 0

    for i, (repo_name, clone_url) in enumerate(repos_to_clone):
        print(f"\nProcessing repository {i+1} of {total_repos}: '{repo_name}' ({clone_url})")
        
        result = clone_repository(repo_name, clone_url, CLONE_DESTINATION_DIR)
        
        if result == "success":
            successful_clones += 1
        elif result == "skipped":
            skipped_clones += 1
        else: # "failed"
            failed_clones += 1
        
        if i < total_repos - 1: # Don't sleep after the last one
            print(f"Waiting for {DELAY_BETWEEN_CLONES} seconds before next attempt...")
            time.sleep(DELAY_BETWEEN_CLONES)
    
    print("\n" + "-" * 40)
    print("--- Cloning Summary ---")
    print(f"Total repositories found for '{ORG_NAME}': {total_repos}")
    print(f"Successfully cloned: {successful_clones}")
    print(f"Already existed (skipped): {skipped_clones}")
    print(f"Failed to clone: {failed_clones}")
    print(f"All repositories from '{ORG_NAME}' attempted. Check logs above for details.")
    print(f"Cloned repositories are in: {os.path.abspath(CLONE_DESTINATION_DIR)}")
    print("--- Script Finished ---")

