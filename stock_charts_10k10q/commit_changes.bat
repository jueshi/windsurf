@echo off
echo Committing changes to stock_charts_10k-10q directory...

REM Initialize Git repository if it doesn't exist
if not exist .git (
    echo Initializing Git repository...
    git init
    if %ERRORLEVEL% neq 0 (
        echo Failed to initialize Git repository.
        exit /b 1
    )
)

REM Configure Git if needed
git config user.name "Stock Charts User"
git config user.email "user@example.com"

REM Add all files
echo Adding files to Git...
git add .
if %ERRORLEVEL% neq 0 (
    echo Failed to add files to Git.
    exit /b 1
)

REM Commit changes
echo Committing changes...
git commit -m "Fix tab switching issue with Extract 10-K Tables button"
if %ERRORLEVEL% neq 0 (
    echo Failed to commit changes.
    exit /b 1
)

echo Changes committed successfully!
