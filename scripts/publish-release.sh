#!/bin/bash
# Publish a new SmartMemory release to PyPI
# Reads version from pyproject.toml automatically
# Usage: ./scripts/publish-release.sh [release-notes]

set -e

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Read version from pyproject.toml (single source of truth)
VERSION=$(grep -m 1 'version = ' pyproject.toml | cut -d'"' -f2)

if [ -z "$VERSION" ]; then
    echo "❌ Could not read version from pyproject.toml"
    exit 1
fi

RELEASE_NOTES=${1:-"Release v$VERSION"}

echo "🚀 Publishing SmartMemory v$VERSION"
echo "   (from pyproject.toml)"
echo ""

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  You have uncommitted changes. Commit them first."
    git status --short
    exit 1
fi

# Verify VERSION file matches pyproject.toml
VERSION_FILE_VERSION=$(cat VERSION | tr -d '\n')
if [ "$VERSION_FILE_VERSION" != "$VERSION" ]; then
    echo "❌ Version mismatch!"
    echo "   pyproject.toml has: $VERSION"
    echo "   VERSION file has: $VERSION_FILE_VERSION"
    echo ""
    echo "Please sync VERSION file with pyproject.toml"
    exit 1
fi

echo "✅ Versions match: $VERSION"
echo ""

# Check if tag already exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "⚠️  Tag v$VERSION already exists"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -d "v$VERSION"
        git push origin ":refs/tags/v$VERSION" 2>/dev/null || true
    else
        exit 1
    fi
fi

# Create git tag
echo "📌 Creating git tag v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION"

# Push tag
echo "📤 Pushing tag to GitHub..."
git push origin "v$VERSION"

echo ""
echo "✅ Tag pushed successfully!"
echo ""

# Check if gh CLI is available
if command -v gh &> /dev/null; then
    echo "📝 Creating GitHub release..."
    
    # Create release with notes
    gh release create "v$VERSION" \
        --title "v$VERSION" \
        --notes "$RELEASE_NOTES" \
        --latest
    
    echo ""
    echo "✅ GitHub release created!"
    echo ""
    echo "🔄 The PyPI publish workflow will start automatically."
    echo "   Monitor progress: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
else
    echo "⚠️  GitHub CLI (gh) not found. Creating release manually..."
    echo ""
    echo "📋 Next steps:"
    echo "   1. Go to: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/releases/new"
    echo "   2. Select tag: v$VERSION"
    echo "   3. Set title: v$VERSION"
    echo "   4. Add release notes"
    echo "   5. Click 'Publish release'"
    echo ""
    echo "   This will trigger the PyPI publish workflow automatically."
fi

echo ""
echo "🎉 Done!"
echo ""
echo "📦 Once published, install with:"
echo "   pip install smartmemory==$VERSION"
