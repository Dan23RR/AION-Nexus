#!/bin/bash

# AION NEXUS - GitHub Repository Setup Script
# This script organizes the repository structure for public release
# Author: Daniel Culotta
# Date: October 2025

echo "================================================"
echo "AION NEXUS v6.5 - Repository Setup"
echo "================================================"

# Create directory structure
echo "Creating directory structure..."

# Documentation directories
mkdir -p docs/business
mkdir -p docs/technical
mkdir -p docs/deployment
mkdir -p docs/api
mkdir -p docs/compliance

# Website directories
mkdir -p website/css
mkdir -p website/js
mkdir -p website/images

# Assets directory
mkdir -p assets/logos
mkdir -p assets/diagrams
mkdir -p assets/screenshots

# GitHub specific directories
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/PULL_REQUEST_TEMPLATE

echo "✓ Directory structure created"

# Copy business documents to docs/business
echo "Organizing business documents..."

# Copy if files exist in parent directory
[ -f ../Executive_Summary.md ] && cp ../Executive_Summary.md docs/business/
[ -f ../INVESTOR_ONE_PAGER.md ] && cp ../INVESTOR_ONE_PAGER.md docs/business/
[ -f ../EMAIL_PRESENTAZIONE_PROGETTO.md ] && cp ../EMAIL_PRESENTAZIONE_PROGETTO.md docs/business/

echo "✓ Business documents organized"

# Copy technical documents to docs/technical
echo "Organizing technical documents..."

# List of technical docs to copy (if they exist)
technical_docs=(
    "ANALISI_TECNICA_COMPLETA.md"
    "AION_NEXUS_EXECUTIVE_SUMMARY.md"
    "CROSS_DOMAIN_VALIDATION_REPORT.md"
    "DEEP_ENSEMBLE_README.md"
    "ADAPTIVE_THRESHOLDS_UPDATE.md"
)

for doc in "${technical_docs[@]}"; do
    [ -f "../$doc" ] && cp "../$doc" "docs/technical/"
done

echo "✓ Technical documents organized"

# Copy website files
echo "Setting up website files..."

# Copy index.html if it exists
[ -f ../index.html ] && cp ../index.html website/

echo "✓ Website files organized"

# Create placeholder files for missing components
echo "Creating placeholder files..."

# Create a simple logo placeholder if needed
if [ ! -f assets/logos/aion-logo.png ]; then
    echo "Note: Add your logo to assets/logos/aion-logo.png"
fi

# Create deployment guide
cat > docs/deployment/quick_start.md << 'EOF'
# AION NEXUS - Quick Start Guide

## Prerequisites
- Industrial facility with critical rotating equipment
- Existing vibration monitoring system (optional but recommended)
- SCADA or MES system for operational data

## Pilot Program (6 months)
1. **Assessment Phase** (Month 1)
2. **Integration Phase** (Month 2-3)
3. **Validation Phase** (Month 4-6)

## Contact
For pilot participation: daniel.culotta@gmail.com
EOF

echo "✓ Placeholder files created"

# Create repository statistics file
echo "Generating repository info..."

cat > REPOSITORY_INFO.md << 'EOF'
# Repository Information

## Structure
```
github_repo/
├── README.md              # Main repository documentation
├── LICENSE                # Apache 2.0 License
├── CONTRIBUTING.md        # Contribution guidelines
├── _config.yml           # GitHub Pages configuration
├── .gitignore            # Git ignore rules
│
├── docs/                 # Documentation
│   ├── business/         # Business documents
│   ├── technical/        # Technical specifications
│   ├── deployment/       # Deployment guides
│   ├── api/             # API documentation
│   └── compliance/       # Regulatory compliance
│
├── website/              # GitHub Pages website
│   ├── index.html       # Main website
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript
│   └── images/          # Website images
│
├── assets/              # Repository assets
│   ├── logos/           # Brand assets
│   ├── diagrams/        # Technical diagrams
│   └── screenshots/     # Application screenshots
│
└── .github/             # GitHub configuration
    ├── workflows/       # GitHub Actions
    └── ISSUE_TEMPLATE/  # Issue templates
```

## Important Notes
- Source code is NOT included (proprietary)
- Models and datasets are NOT included (proprietary)
- This repository contains documentation and website only
- For source code licensing: contact daniel.culotta@gmail.com
EOF

echo "✓ Repository info generated"

# Create summary
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Review the organized structure in github_repo/"
echo "2. Add your logo to assets/logos/aion-logo.png"
echo "3. Copy your index.html to website/"
echo "4. Initialize git repository:"
echo "   cd github_repo"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit: AION NEXUS v6.5 documentation'"
echo ""
echo "5. Create GitHub repository at github.com/new"
echo "6. Push to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/aion-nexus.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "7. Enable GitHub Pages:"
echo "   - Go to Settings → Pages"
echo "   - Source: Deploy from branch"
echo "   - Branch: main, folder: / (root)"
echo ""
echo "Repository is ready for professional presentation!"